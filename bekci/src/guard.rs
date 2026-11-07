//! Karar noktası: bir araç çağrısı ya izin alır, ya onaya bağlanır, ya reddedilir. Her karar actions.log'a yazılır.
//!
//! - Çağrı tam olarak `{"name", "arguments"}` olmalı; araç beyaz listede, argümanlar şemaya birebir uymalı
//!   (eksik zorunlu, fazladan alan, yanlış tür → ret; `true` sayı değildir, `1.0` tamsayı değildir).
//! - Onaylı çağrıda argümanlar bilette saklanır: onaydan sonra değiştirilemez, bilet tek kullanımlıktır, süresi dolar.
//! - Güvenilmez çıktılı bir aracın (dosya, web, pano) sonucu geldikten sonra, kullanıcı yeni bir talimat verene dek
//!   (`turn`) yan etkili her araç onaya yükseltilir: veri talimat değildir.
//! - Acil durdurma: bekleyen onaylar ve süren işler iptal edilir; `resume`e dek her çağrı reddedilir.
//! - Oturum ağ izni (`network_session`): salt okunur bir ağ aracının onayında kullanıcı "bu oturumda sorma" da
//!   diyebilir; sonraki salt okunur ağ çağrıları sorulmadan geçer (yine kaydedilir). Konuşmada güvenilmez veri varken
//!   (bu turda ya da istemcinin bildirdiği geçmişte) izin askıdadır: veri, sorgu metni üzerinden dışarı taşınamasın.
//!   Kısıtın değiştirdiği çağrı izni kullanamaz; acil durdurma izni kaldırır.

use crate::log::ActionLog;
use crate::policy::{Danger, Kind, Policy, Tool};
use crate::validate;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::time::Instant;

#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// `via`: onaysız geçişin özel nedeni (oturum ağ izni)
    Allow { exec: u64, tool: String, args: Map<String, Value>, via: Option<String> },
    /// `session`: "offer" (onayla birlikte oturum izni verilebilir) | "suspended" (izin var ama konuşma kirli)
    Confirm { ticket: String, tool: String, args: Map<String, Value>, summary: String, reason: String, session: Option<String> },
    Deny { reason: String },
}

impl Outcome {
    pub fn to_json(&self) -> Value {
        match self {
            Outcome::Allow { exec, tool, args, via } => {
                let mut v = json!({"decision": "allow", "exec": exec, "tool": tool, "args": args});
                if let Some(x) = via {
                    v["via"] = json!(x);
                }
                v
            }
            Outcome::Confirm { ticket, tool, args, summary, reason, session } => {
                let mut v = json!({"decision": "confirm", "ticket": ticket, "tool": tool, "args": args,
                                   "summary": summary, "reason": reason});
                if let Some(x) = session {
                    v["session"] = json!(x);
                }
                v
            }
            Outcome::Deny { reason } => json!({"decision": "deny", "reason": reason}),
        }
    }
}

struct Pending {
    tool: String,
    args: Map<String, Value>,
    created: Instant,
    offer: bool, // onayla birlikte oturum ağ izni verilebilir
}

pub struct Guard {
    pub policy: Policy,
    log: ActionLog,
    pending: HashMap<String, Pending>,
    running: HashMap<u64, String>,
    next_exec: u64,
    stopped: bool,
    tainted: bool,
    history_tainted: bool, // istemci: konuşma geçmişinde hâlâ güvenilmez içerik var (turn ile bildirilir)
    net_grant: bool,
}

fn fields(pairs: &[(&str, Value)]) -> Map<String, Value> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
}

/// Argümanları aracın şemasına göre doğrular ve normalleştirilmiş kopyasını döndürür.
pub fn check_args(tool: &Tool, args: &Value, policy: &Policy) -> Result<Map<String, Value>, String> {
    let obj = args.as_object().ok_or("arguments bir nesne olmalı")?;
    if let Some(k) = obj.keys().find(|k| !tool.params.iter().any(|p| &p.name == *k)) {
        return Err(format!("bilinmeyen parametre: {}", validate::display(k)));
    }
    let mut out = Map::new();
    for p in &tool.params {
        let Some(v) = obj.get(&p.name) else {
            if p.required {
                return Err(format!("eksik zorunlu parametre: {}", p.name));
            }
            continue;
        };
        let bad = |e: String| format!("{}: {e}", p.name);
        let norm = match &p.kind {
            Kind::Text { max_len, multiline } => {
                let s = v.as_str().ok_or_else(|| bad("metin olmalı".into()))?;
                Value::String(validate::check_text(s, *max_len, *multiline).map_err(bad)?)
            }
            Kind::Shell { max_len } => {
                let s = v.as_str().ok_or_else(|| bad("metin olmalı".into()))?;
                Value::String(validate::check_text(s, *max_len, false).map_err(bad)?)
            }
            Kind::Name { max_len } => {
                let s = v.as_str().ok_or_else(|| bad("metin olmalı".into()))?;
                Value::String(validate::check_name(s, *max_len).map_err(bad)?)
            }
            Kind::Integer { min, max } => {
                let n = match v {
                    Value::Number(n) if !n.is_f64() => n.as_i64().ok_or_else(|| bad("aralık dışı".into()))?,
                    _ => return Err(bad("tamsayı olmalı".into())),
                };
                if n < *min || n > *max {
                    return Err(bad(format!("{min}..{max} aralığında olmalı")));
                }
                json!(n)
            }
            Kind::Number { min, max } => {
                let x = match v {
                    Value::Number(n) => n.as_f64().ok_or_else(|| bad("sayı olmalı".into()))?,
                    _ => return Err(bad("sayı olmalı".into())),
                };
                if x < *min || x > *max {
                    return Err(bad(format!("{min}..{max} aralığında olmalı")));
                }
                json!(x)
            }
            Kind::Boolean => Value::Bool(v.as_bool().ok_or_else(|| bad("true/false olmalı".into()))?),
            Kind::Enum(vals) => {
                let s = v.as_str().ok_or_else(|| bad("metin olmalı".into()))?;
                if !vals.iter().any(|x| x == s) {
                    return Err(bad(format!("izinli değerlerden biri olmalı: {}", vals.join(", "))));
                }
                Value::String(s.to_owned())
            }
            Kind::Path => {
                let s = v.as_str().ok_or_else(|| bad("yol metin olmalı".into()))?;
                Value::String(validate::check_path(s, &policy.roots, &policy.deny).map_err(bad)?)
            }
            Kind::Url => {
                let s = v.as_str().ok_or_else(|| bad("adres metin olmalı".into()))?;
                Value::String(validate::check_url(s).map_err(bad)?)
            }
        };
        out.insert(p.name.clone(), norm);
    }
    Ok(out)
}

/// Onay penceresinde gösterilecek, görünmez karakterleri kaçışlanmış özet.
pub fn summary(tool: &Tool, args: &Map<String, Value>) -> String {
    let mut s = format!("{}\n{}", tool.name, validate::display(&tool.desc));
    for p in &tool.params {
        if let Some(v) = args.get(&p.name) {
            let shown = match v {
                Value::String(x) => validate::display(x),
                o => o.to_string(),
            };
            s.push_str(&format!("\n  {}: {}", p.name, shown));
        }
    }
    s
}

impl Guard {
    pub fn new(policy: Policy) -> Result<Guard, String> {
        let log = ActionLog::open(&policy.log)?;
        let mut g = Guard { policy, log, pending: HashMap::new(), running: HashMap::new(), next_exec: 1, stopped: false,
                           tainted: false, history_tainted: false, net_grant: false };
        let tools: Vec<&str> = g.policy.tools.iter().map(|t| t.name.as_str()).collect();
        let f = fields(&[("tools", json!(tools)), ("version", json!(env!("CARGO_PKG_VERSION")))]);
        g.log.write("start", f)?;
        Ok(g)
    }

    pub fn log_event(&mut self, event: &str, f: &[(&str, Value)]) -> Result<(), String> {
        self.log.write(event, fields(f))
    }

    pub fn is_stopped(&self) -> bool {
        self.stopped
    }
    pub fn is_tainted(&self) -> bool {
        self.tainted
    }
    pub fn has_net_grant(&self) -> bool {
        self.net_grant
    }
    pub fn log_path(&self) -> std::path::PathBuf {
        self.log.path().to_path_buf()
    }

    /// Kaydı yazılamayan karar reddedilir.
    fn logged(&mut self, event: &str, f: Map<String, Value>, out: Outcome) -> Outcome {
        let mut f = f;
        for (k, v) in out.to_json().as_object().into_iter().flatten() {
            f.insert(k.clone(), v.clone());
        }
        match self.log.write(event, f) {
            Ok(()) => out,
            Err(e) => {
                self.pending.clear();
                Outcome::Deny { reason: format!("kayıt yazılamadı, işlem reddedildi: {e}") }
            }
        }
    }

    /// Modelin önerdiği tek bir araç çağrısını denetler.
    pub fn check(&mut self, call: &Value) -> Outcome {
        self.check_with(call, false)
    }

    /// `altered`: motorun kısıtlı çözümlemesi bu çağrının içinde modelin en olası token'ını reddedip başkasını seçti
    /// (ör. model 150 yazacaktı, aralık 0..100 olduğu için 15'te kapandı). Değer modelin niyetinden farklı olabilir.
    pub fn check_with(&mut self, call: &Value, altered: bool) -> Outcome {
        let mut base = fields(&[("call", call.clone()), ("tainted", json!(self.tainted))]);
        if altered {
            base.insert("altered".into(), json!(true));
        }
        let out = self.decide(call, altered);
        self.logged("check", base, out)
    }

    fn decide(&mut self, call: &Value, altered: bool) -> Outcome {
        let deny = |r: String| Outcome::Deny { reason: r };
        if self.stopped {
            return deny("acil durdurma etkin; kullanıcı devam ettirene dek araç çalışmaz".into());
        }
        let Some(obj) = call.as_object() else { return deny("çağrı bir nesne olmalı".into()) };
        if let Some(k) = obj.keys().find(|k| *k != "name" && *k != "arguments") {
            return deny(format!("çağrıda bilinmeyen alan: {}", validate::display(k)));
        }
        let Some(name) = obj.get("name").and_then(Value::as_str) else { return deny("araç adı (name) eksik".into()) };
        let Some(tool) = self.policy.tool(name).cloned() else {
            return deny(format!("beyaz listede olmayan araç: {}", validate::display(name)));
        };
        if tool.danger == Danger::Blocked {
            return deny(format!("engelli araç: {name}"));
        }
        let empty = Value::Object(Map::new());
        let args = match check_args(&tool, obj.get("arguments").unwrap_or(&empty), &self.policy) {
            Ok(a) => a,
            Err(e) => return deny(e),
        };
        // Oturum ağ izni yalnız salt okunur ağ araçlarında ve kısıtın değiştirmediği çağrıda geçerlidir.
        let session_ok = self.policy.network_session && tool.network && tool.readonly && !altered;
        let clean = !self.tainted && !self.history_tainted;
        if session_ok && clean && self.net_grant && tool.danger == Danger::Confirm {
            return self.allow(tool.name, args, Some("session_grant"));
        }
        let session = match (session_ok && tool.danger == Danger::Confirm, clean, self.net_grant) {
            (true, true, _) => Some("offer"),
            (true, false, true) => Some("suspended"),
            _ => None,
        };
        let reason = if tool.danger == Danger::Confirm {
            Some("araç her çağrıda onay ister")
        } else if self.tainted && !tool.readonly {
            Some("konuşmada güvenilmez veri var (dosya/web/pano); yan etkili araç onaya yükseltildi")
        } else if altered && !tool.readonly {
            Some("kısıtlı çözümleme modelin seçimini değiştirdi; değer istenenden farklı olabilir, onaya yükseltildi")
        } else {
            None
        };
        match reason {
            Some(r) => {
                let ticket = match crate::sys::random_ticket() {
                    Ok(t) => t,
                    Err(e) => return deny(format!("onay bileti üretilemedi: {e}")),
                };
                let summary = summary(&tool, &args);
                let offer = session == Some("offer");
                self.pending.insert(ticket.clone(), Pending { tool: tool.name.clone(), args: args.clone(), created: Instant::now(), offer });
                Outcome::Confirm { ticket, tool: tool.name, args, summary, reason: r.into(), session: session.map(Into::into) }
            }
            None => self.allow(tool.name, args, None),
        }
    }

    fn allow(&mut self, tool: String, args: Map<String, Value>, via: Option<&str>) -> Outcome {
        let exec = self.next_exec;
        self.next_exec += 1;
        self.running.insert(exec, tool.clone());
        Outcome::Allow { exec, tool, args, via: via.map(Into::into) }
    }

    /// Kullanıcının onay yanıtı. Bilet tek kullanımlık; süresi dolmuşsa ya da durdurma etkinse ret.
    pub fn confirm(&mut self, ticket: &str, yes: bool, via: &str) -> Outcome {
        self.confirm_with(ticket, yes, via, false)
    }

    /// `session`: kullanıcı onayla birlikte oturum ağ iznini de verdi (yalnız biletin teklifi varsa geçerli).
    pub fn confirm_with(&mut self, ticket: &str, yes: bool, via: &str, session: bool) -> Outcome {
        let mut base = fields(&[("ticket", json!(ticket)), ("answer", json!(if yes { "yes" } else { "no" })), ("via", json!(via))]);
        let grant = yes && session && !self.stopped && self.pending.get(ticket).is_some_and(|p| p.offer);
        let out = match self.pending.remove(ticket) {
            None => Outcome::Deny { reason: "bilinmeyen ya da kullanılmış onay bileti".into() },
            Some(_) if self.stopped => Outcome::Deny { reason: "acil durdurma etkin".into() },
            Some(p) if p.created.elapsed() > self.policy.confirm_timeout => {
                Outcome::Deny { reason: format!("onay zaman aşımı ({} sn): yanıt verilmedi sayıldı", self.policy.confirm_timeout.as_secs()) }
            }
            Some(_) if !yes => Outcome::Deny { reason: "kullanıcı reddetti".into() },
            Some(p) => self.allow(p.tool, p.args, None),
        };
        if grant && matches!(out, Outcome::Allow { .. }) {
            self.net_grant = true;
            base.insert("net_grant".into(), json!(true));
        }
        self.logged("confirm", base, out)
    }

    /// Süresi dolan biletleri düşürür (zaman aşımı = Hayır) ve düşenleri döndürür.
    pub fn expire(&mut self) -> Vec<String> {
        let t = self.policy.confirm_timeout;
        let gone: Vec<String> = self.pending.iter().filter(|(_, p)| p.created.elapsed() > t).map(|(k, _)| k.clone()).collect();
        for k in &gone {
            self.pending.remove(k);
            let _ = self.log.write("expire", fields(&[("ticket", json!(k))]));
        }
        gone
    }

    pub fn is_pending(&self, ticket: &str) -> bool {
        self.pending.contains_key(ticket)
    }

    /// Çalıştırıcının sonuç bildirimi. Güvenilmez çıktılı aracın sonucu (başarısız olsa da: hata metni içerikten parça
    /// taşıyabilir) oturumu kirletir.
    pub fn result(&mut self, exec: u64, ok: bool, detail: Option<&str>) -> Result<Value, String> {
        let tool = self.running.remove(&exec).ok_or("bilinmeyen ya da zaten bildirilmiş exec")?;
        let untrusted = self.policy.tool(&tool).is_some_and(|t| t.untrusted);
        if untrusted {
            self.tainted = true;
        }
        let d = detail.map(|s| validate::display(&s.chars().take(500).collect::<String>()));
        self.log.write("result", fields(&[("exec", json!(exec)), ("tool", json!(tool)), ("ok", json!(ok)), ("detail", json!(d)), ("tainted", json!(self.tainted))]))?;
        Ok(json!({"ok": true, "tainted": self.tainted}))
    }

    /// Kullanıcının yeni talimatı: güvenilmez veri bayrağı kalkar.
    pub fn turn(&mut self) -> Result<Value, String> {
        self.turn_with(false)
    }

    /// `untrusted_history`: istemcinin konuşma geçmişinde hâlâ güvenilmez araç çıktısı var (oturum ağ izni askıda kalır;
    /// yan etkili araçların onaya yükselmesi ise yalnız bu turdaki içeriğe bağlıdır).
    pub fn turn_with(&mut self, untrusted_history: bool) -> Result<Value, String> {
        self.tainted = false;
        self.history_tainted = untrusted_history;
        self.log.write("turn", fields(&[("untrusted_history", json!(untrusted_history))]))?;
        Ok(json!({"ok": true}))
    }

    /// Acil durdurma: bekleyen onaylar düşer, süren işlerin kimlikleri iptal için döndürülür.
    pub fn stop(&mut self, origin: &str) -> Value {
        self.stopped = true;
        let revoked = std::mem::take(&mut self.net_grant);
        let tickets: Vec<String> = self.pending.drain().map(|(k, _)| k).collect();
        let mut cancelled: Vec<u64> = self.running.drain().map(|(k, _)| k).collect();
        cancelled.sort_unstable();
        let _ = self.log.write("stop", fields(&[("origin", json!(origin)), ("cancelled", json!(cancelled)), ("tickets", json!(tickets)),
                                                ("net_grant_revoked", json!(revoked))]));
        json!({"ok": true, "stopped": true, "cancelled": cancelled, "tickets": tickets})
    }

    pub fn resume(&mut self, origin: &str) -> Result<Value, String> {
        self.stopped = false;
        self.log.write("resume", fields(&[("origin", json!(origin))]))?;
        Ok(json!({"ok": true, "stopped": false}))
    }
}
