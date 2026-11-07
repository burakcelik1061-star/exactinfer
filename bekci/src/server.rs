//! JSON satır protokolü. Her istek bir satır, her yanıt bir satır; ayrıca kendiliğinden olay satırları
//! (`{"event": ...}`): acil durdurma kısayolu, onay penceresi yanıtı, onay zaman aşımı.
//!
//! İşlemler (`op`): hello, tools, parse {text}, check {call}, confirm {ticket, answer: "yes"|"no", session?},
//! result {exec, ok, detail?}, turn {untrusted_history?}, stop, resume, quit. İsteğe konan `id` yanıtta aynen döner.

use crate::calls::extract_calls;
use crate::guard::{Guard, Outcome};
use crate::policy::ConfirmUi;
use crate::sys;
use serde_json::{json, Value};
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub type Out = Arc<Mutex<Box<dyn Write + Send>>>;

#[derive(Clone)]
pub struct Server {
    pub guard: Arc<Mutex<Guard>>,
    out: Out,
    ui: ConfirmUi,
}

fn dialog_title(ticket: &str) -> String {
    format!("Bekçi onayı · {}", &ticket[..8])
}

impl Server {
    pub fn new(guard: Guard, out: Out) -> Server {
        let ui = guard.policy.confirm_ui;
        Server { guard: Arc::new(Mutex::new(guard)), out, ui }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Guard> {
        self.guard.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn emit(&self, v: &Value) {
        let mut o = self.out.lock().unwrap_or_else(|p| p.into_inner());
        let _ = writeln!(o, "{v}");
        let _ = o.flush();
    }

    /// Diyalog modunda onay penceresini ayrı iş parçacığında açar; yanıt olay olarak bildirilir.
    fn after_check(&self, out: &Outcome) {
        if let (ConfirmUi::Dialog, Outcome::Confirm { ticket, summary, .. }) = (self.ui, out) {
            let (s, ticket, text) = (self.clone(), ticket.clone(), summary.clone());
            std::thread::spawn(move || {
                let text = format!("Asistan şu işlemi yapmak istiyor:\n\n{text}\n\nOnaylıyor musunuz?");
                let yes = sys::confirm_dialog(&dialog_title(&ticket), &text);
                let mut g = s.lock();
                if !g.is_pending(&ticket) {
                    return; // zaman aşımı ya da acil durdurma zaten karar verdi
                }
                let mut ev = g.confirm(&ticket, yes, "dialog").to_json();
                drop(g);
                ev["event"] = json!("decision");
                ev["ticket"] = json!(s.public_ticket(&ticket));
                s.emit(&ev);
            });
        }
    }

    /// Diyalog modunda istemci bileti onaylayamaz; olayları eşleştirmek için yalnızca ilk 8 hanesini görür.
    fn public_ticket(&self, t: &str) -> String {
        if self.ui == ConfirmUi::Dialog { t[..8].to_string() } else { t.to_string() }
    }

    fn dismiss(&self, tickets: &[String]) {
        if self.ui == ConfirmUi::Dialog {
            for t in tickets {
                sys::dismiss_dialog(&dialog_title(t));
            }
        }
    }

    /// Acil durdurma (kısayol, sesli komut ya da istemci): olay olarak da duyurulur.
    pub fn stop(&self, origin: &str) -> Value {
        let r = self.lock().stop(origin);
        let tickets: Vec<String> = serde_json::from_value(r["tickets"].clone()).unwrap_or_default();
        self.dismiss(&tickets);
        r
    }

    /// Süresi dolan onayları düşürür (zaman aşımı = Hayır); saniyede bir çağrılır.
    pub fn tick(&self) {
        let gone = self.lock().expire();
        self.dismiss(&gone);
        for t in gone {
            self.emit(&json!({"event": "decision", "ticket": self.public_ticket(&t), "decision": "deny", "reason": "onay zaman aşımı: yanıt verilmedi sayıldı"}));
        }
    }

    pub fn spawn_ticker(&self) {
        let s = self.clone();
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_millis(500));
            s.tick();
        });
    }

    /// Bir istek satırını işler; (yanıt, çıkılsın mı).
    pub fn handle_line(&self, line: &str) -> (Value, bool) {
        let req = match crate::jsonx::parse_strict(line) {
            Ok(Value::Object(m)) => m,
            Ok(_) => return (self.bad(None, "istek bir JSON nesnesi olmalı", line), false),
            Err(e) => return (self.bad(None, &e, line), false),
        };
        let id = req.get("id").filter(|v| v.is_string() || v.is_u64()).cloned();
        let op = req.get("op").and_then(Value::as_str).unwrap_or("");
        let str_field = |k: &str| req.get(k).and_then(Value::as_str);
        let (mut resp, quit) = match op {
            "hello" => {
                let g = self.lock();
                (json!({"ok": true, "name": "bekci", "version": env!("CARGO_PKG_VERSION"),
                        "confirm_ui": if self.ui == ConfirmUi::Dialog { "dialog" } else { "client" },
                        "stopped": g.is_stopped(), "tainted": g.is_tainted(), "net_grant": g.has_net_grant(),
                        "network_session": g.policy.network_session, "log": g.log_path()}), false)
            }
            "tools" => (json!({"ok": true, "tools": self.lock().policy.tools_schema()}), false),
            "check" => match req.get("call") {
                Some(c) => {
                    let altered = match req.get("altered") {
                        None => false,
                        Some(Value::Bool(b)) => *b,
                        Some(_) => return (self.bad(id, "check: 'altered' true/false olmalı", line), false),
                    };
                    let out = self.lock().check_with(c, altered);
                    self.after_check(&out);
                    (self.with_via(out), false)
                }
                None => return (self.bad(id, "check: 'call' eksik", line), false),
            },
            "parse" => match str_field("text") {
                Some(text) => match extract_calls(text) {
                    Ok(calls) => {
                        // altered: motorun kısıtla değiştirdiği çağrıların sıra numaraları (0'dan)
                        let altered: Vec<u64> = match req.get("altered") {
                            None => Vec::new(),
                            Some(Value::Array(a)) if a.iter().all(|v| v.as_u64().is_some_and(|i| i < calls.len() as u64)) => {
                                a.iter().filter_map(Value::as_u64).collect()
                            }
                            Some(_) => return (self.bad(id, "parse: 'altered' çağrı sıra numaralarından oluşan dizi olmalı", line), false),
                        };
                        let mut res = Vec::new();
                        for (i, c) in calls.iter().enumerate() {
                            let out = self.lock().check_with(c, altered.contains(&(i as u64)));
                            self.after_check(&out);
                            res.push(self.with_via(out));
                        }
                        (json!({"ok": true, "calls": res}), false)
                    }
                    Err(e) => {
                        let _ = self.lock().log_event("parse_error", &[("error", json!(e)), ("text", json!(text.chars().take(2000).collect::<String>()))]);
                        (json!({"ok": false, "error": format!("model çıktısı reddedildi: {e}"), "calls": []}), false)
                    }
                },
                None => return (self.bad(id, "parse: 'text' eksik", line), false),
            },
            "confirm" => {
                if self.ui == ConfirmUi::Dialog {
                    return (self.bad(id, "onay yalnızca bekçinin kendi penceresinden verilir (confirm_ui = dialog)", line), false);
                }
                let answer = match str_field("answer") {
                    Some("yes") => true,
                    Some("no") => false,
                    _ => return (self.bad(id, "confirm: answer \"yes\" ya da \"no\" olmalı", line), false),
                };
                let session = match req.get("session") {
                    None => false,
                    Some(Value::Bool(b)) => *b,
                    Some(_) => return (self.bad(id, "confirm: 'session' true/false olmalı", line), false),
                };
                match str_field("ticket") {
                    Some(t) => {
                        let mut g = self.lock();
                        let before = g.has_net_grant();
                        let mut v = g.confirm_with(t, answer, "client", session).to_json();
                        v["net_grant"] = json!(g.has_net_grant() && !before);
                        (v, false)
                    }
                    None => return (self.bad(id, "confirm: 'ticket' eksik", line), false),
                }
            }
            "result" => {
                let (Some(exec), Some(ok)) = (req.get("exec").and_then(Value::as_u64), req.get("ok").and_then(Value::as_bool)) else {
                    return (self.bad(id, "result: 'exec' (sayı) ve 'ok' (true/false) gerekli", line), false);
                };
                let r = self.lock().result(exec, ok, str_field("detail"));
                (r.unwrap_or_else(|e| json!({"ok": false, "error": e})), false)
            }
            "turn" => {
                let hist = match req.get("untrusted_history") {
                    None => false,
                    Some(Value::Bool(b)) => *b,
                    Some(_) => return (self.bad(id, "turn: 'untrusted_history' true/false olmalı", line), false),
                };
                (self.lock().turn_with(hist).unwrap_or_else(|e| json!({"ok": false, "error": e})), false)
            }
            "stop" => (self.stop(str_field("origin").unwrap_or("client")), false),
            "resume" => (self.lock().resume("client").unwrap_or_else(|e| json!({"ok": false, "error": e})), false),
            "quit" => {
                let _ = self.lock().log_event("quit", &[]);
                (json!({"ok": true}), true)
            }
            _ => return (self.bad(id, "bilinmeyen işlem (op)", line), false),
        };
        if let Some(i) = id {
            resp["id"] = i;
        }
        (resp, quit)
    }

    fn with_via(&self, out: Outcome) -> Value {
        let mut v = out.to_json();
        if matches!(out, Outcome::Confirm { .. }) {
            v["via"] = json!(if self.ui == ConfirmUi::Dialog { "dialog" } else { "client" });
            v["ticket"] = json!(v["ticket"].as_str().map(|t| self.public_ticket(t)));
        }
        v
    }

    fn bad(&self, id: Option<Value>, err: &str, line: &str) -> Value {
        let snippet: String = line.chars().take(300).collect();
        let _ = self.lock().log_event("bad_request", &[("error", json!(err)), ("line", json!(snippet))]);
        let mut v = json!({"ok": false, "error": err});
        if let Some(i) = id {
            v["id"] = i;
        }
        v
    }
}
