//! Gerçek `bekci serve` süreci üzerinden JSON satır protokolü.

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

struct Proc {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    dir: PathBuf,
}

impl Proc {
    fn start(name: &str, ui: &str) -> Proc {
        let dir = std::env::temp_dir().join(format!("bekci-ipc-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let root = dir.join("izinli");
        std::fs::create_dir_all(&root).unwrap();
        let policy = dir.join("politika").join("bekci.toml");
        std::fs::create_dir_all(policy.parent().unwrap()).unwrap();
        let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/policy.example.toml"))
            .unwrap()
            .replace("# log = \"%LOCALAPPDATA%\\\\engine-lab\\\\actions.log\"   # varsayılan",
                     &format!("log = '{}'", dir.join("veri").join("actions.log").display()))
            .replace("confirm_ui = \"dialog\"", &format!("confirm_ui = \"{ui}\""))
            .replace("confirm_timeout_s = 60", "confirm_timeout_s = 2")
            .replace("hotkey = \"ctrl+alt+end\"", "")
            .replace("roots = [\"%USERPROFILE%\\\\Documents\", \"%USERPROFILE%\\\\Desktop\", \"%USERPROFILE%\\\\Downloads\"]",
                     &format!("roots = ['{}']", dir.display()));
        std::fs::write(&policy, text).unwrap();
        let mut child = Command::new(env!("CARGO_BIN_EXE_bekci"))
            .args(["serve", "--policy", &policy.display().to_string()])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Proc { child, stdin, stdout, dir }
    }

    fn send_raw(&mut self, line: &str) {
        self.stdin.write_all(line.as_bytes()).unwrap();
        self.stdin.write_all(b"\n").unwrap();
        self.stdin.flush().unwrap();
    }

    fn read(&mut self) -> Value {
        let mut s = String::new();
        self.stdout.read_line(&mut s).unwrap();
        serde_json::from_str(&s).unwrap_or_else(|e| panic!("yanıt JSON değil ({e}): {s:?}"))
    }

    fn req(&mut self, v: Value) -> Value {
        self.send_raw(&v.to_string());
        self.read()
    }

    fn log(&self) -> Vec<Value> {
        std::fs::read_to_string(self.dir.join("veri").join("actions.log"))
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }
}

impl Drop for Proc {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

#[test]
fn protocol_round_trip() {
    let mut p = Proc::start("rt", "client");
    let h = p.req(json!({"op": "hello", "id": "a1"}));
    assert_eq!(h["ok"], true);
    assert_eq!(h["id"], "a1");
    assert_eq!(h["confirm_ui"], "client");

    let tools = p.req(json!({"op": "tools"}));
    let names: Vec<&str> = tools["tools"].as_array().unwrap().iter().map(|t| t["function"]["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"open_app") && !names.contains(&"shutdown"), "engelli araç modele gösterilmez: {names:?}");

    // modelin ham çıktısı (Qwen3 biçimi)
    let text = "Not defterini açıyorum.\n<tool_call>\n{\"name\": \"open_app\", \"arguments\": {\"name\": \"Not Defteri\"}}\n</tool_call>\n\
                <tool_call>\n{\"name\": \"shutdown\", \"arguments\": {}}\n</tool_call>";
    let r = p.req(json!({"op": "parse", "text": text, "id": 7}));
    assert_eq!(r["id"], 7);
    assert_eq!(r["calls"][0]["decision"], "allow");
    assert_eq!(r["calls"][1]["decision"], "deny");
    let exec = r["calls"][0]["exec"].as_u64().unwrap();
    assert_eq!(p.req(json!({"op": "result", "exec": exec, "ok": true}))["ok"], true);

    let bad = p.req(json!({"op": "parse", "text": "<tool_call>{\"name\":\"open_app\",\"name\":\"shutdown\",\"arguments\":{}}</tool_call>"}));
    assert_eq!(bad["ok"], false);
    assert!(bad["error"].as_str().unwrap().contains("yinelenen"));

    // onay akışı (istemci modu)
    let path = format!("{}\\izinli\\eski.txt", p.dir.display());
    let c = p.req(json!({"op": "check", "call": {"name": "file_delete", "arguments": {"path": path}}}));
    assert_eq!(c["decision"], "confirm");
    assert_eq!(c["via"], "client");
    let t = c["ticket"].as_str().unwrap().to_string();
    assert_eq!(p.req(json!({"op": "confirm", "ticket": t, "answer": "evet"}))["ok"], false, "yalnız yes/no");
    let y = p.req(json!({"op": "confirm", "ticket": t, "answer": "yes"}));
    assert_eq!(y["decision"], "allow");
    assert_eq!(p.req(json!({"op": "confirm", "ticket": t, "answer": "yes"}))["decision"], "deny");

    // yanıtsız onay: zaman aşımı olayı gelir
    let c2 = p.req(json!({"op": "check", "call": {"name": "open_url", "arguments": {"url": "https://example.com"}}}));
    let ev = p.read();
    assert_eq!(ev["event"], "decision");
    assert_eq!(ev["ticket"], c2["ticket"]);
    assert_eq!(ev["decision"], "deny");

    // acil durdurma
    let s = p.req(json!({"op": "stop", "origin": "ses:dur"}));
    assert_eq!(s["stopped"], true);
    assert_eq!(p.req(json!({"op": "check", "call": {"name": "system_status", "arguments": {}}}))["decision"], "deny");
    p.req(json!({"op": "resume"}));
    assert_eq!(p.req(json!({"op": "check", "call": {"name": "system_status", "arguments": {}}}))["decision"], "allow");

    // kısıtlı çözümlemenin değiştirdiği çağrı işareti (salt okunur araçta karar değişmez)
    let tc = "<tool_call>{\"name\": \"system_status\", \"arguments\": {}}</tool_call>";
    assert_eq!(p.req(json!({"op": "parse", "text": tc, "altered": [0]}))["calls"][0]["decision"], "allow");
    assert_eq!(p.req(json!({"op": "parse", "text": tc, "altered": [1]}))["ok"], false, "olmayan çağrı numarası");
    assert_eq!(p.req(json!({"op": "check", "call": {"name": "system_status"}, "altered": "evet"}))["ok"], false);

    // bozuk istekler süreci düşürmez
    p.send_raw("{bozuk");
    assert_eq!(p.read()["ok"], false);
    p.send_raw(&"x".repeat(300 * 1024));
    assert!(p.read()["error"].as_str().unwrap().contains("çok uzun"));
    assert_eq!(p.req(json!({"op": "sil_her_seyi"}))["ok"], false);
    assert_eq!(p.req(json!({"op": "hello"}))["ok"], true);

    assert_eq!(p.req(json!({"op": "quit"}))["ok"], true);
    assert!(p.child.wait().unwrap().success());
    let events: Vec<String> = p.log().iter().map(|v| v["event"].as_str().unwrap().to_string()).collect();
    for e in ["start", "check", "result", "parse_error", "confirm", "expire", "stop", "resume", "bad_request", "quit", "exit"] {
        assert!(events.iter().any(|x| x == e), "günlükte '{e}' yok: {events:?}");
    }
}

#[test]
fn model_cannot_touch_policy_or_log() {
    let mut p = Proc::start("self", "client");
    let pol = format!("{}\\politika\\bekci.toml", p.dir.display());
    let log = format!("{}\\veri\\actions.log", p.dir.display());
    for path in [pol, log] {
        let r = p.req(json!({"op": "check", "call": {"name": "file_read", "arguments": {"path": path}}}));
        assert_eq!(r["decision"], "deny", "{r}");
        assert!(r["reason"].as_str().unwrap().contains("korumalı"));
    }
}

#[test]
fn dialog_mode_client_cannot_confirm() {
    // Pencere açılır ve 2 sn içinde yanıtlanmazsa bekçi kendisi "Hayır"a basar.
    if std::env::var("BEKCI_DIALOG_TEST").is_err() {
        eprintln!("ekranda pencere açtığı için atlandı (BEKCI_DIALOG_TEST=1 ile çalışır)");
        return;
    }
    let mut p = Proc::start("dialog", "dialog");
    let path = format!("{}\\izinli\\x.txt", p.dir.display());
    let c = p.req(json!({"op": "check", "call": {"name": "file_delete", "arguments": {"path": path}}}));
    assert_eq!(c["via"], "dialog");
    let short = c["ticket"].as_str().unwrap().to_string();
    assert_eq!(short.len(), 8, "istemci tam bileti görmez");
    assert_eq!(p.req(json!({"op": "confirm", "ticket": short, "answer": "yes"}))["ok"], false);
    let ev = p.read();
    assert_eq!(ev["event"], "decision");
    assert_eq!(ev["ticket"], short);
    assert_eq!(ev["decision"], "deny");
}
