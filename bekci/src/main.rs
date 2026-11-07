//! bekci serve --policy <dosya>      JSON satır protokolüyle çalışır (stdin istek, stdout yanıt/olay)
//! bekci check-policy <dosya>        politikayı doğrular, araç şemasını yazdırır
//!
//! Yönetici yetkisiyle başlatılırsa çalışmayı reddeder (çıkış kodu 3).

use bekci::guard::Guard;
use bekci::policy::Policy;
use bekci::server::{Out, Server};
use serde_json::json;
use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const MAX_LINE: usize = 256 * 1024;

fn usage() -> ! {
    eprintln!("kullanım:\n  bekci serve --policy <politika.toml>\n  bekci check-policy <politika.toml>");
    std::process::exit(2)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args.first().map(String::as_str) {
        Some("check-policy") if args.len() == 2 => match Policy::load(&PathBuf::from(&args[1])) {
            Ok(p) => {
                println!("{}", serde_json::to_string_pretty(&p.tools_schema()).unwrap());
                eprintln!("politika geçerli: {} araç, {} izinli kök, günlük {}", p.tools.len(), p.roots.len(), p.log.display());
                0
            }
            Err(e) => {
                eprintln!("politika hatalı: {e}");
                1
            }
        },
        Some("serve") if args.len() == 3 && args[1] == "--policy" => serve(&PathBuf::from(&args[2])),
        _ => usage(),
    };
    std::process::exit(code)
}

fn serve(policy_path: &Path) -> i32 {
    if bekci::sys::is_admin() {
        eprintln!("bekçi yönetici yetkisiyle çalışmaz; normal kullanıcı olarak başlatın");
        return 3;
    }
    let policy = match Policy::load(policy_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("politika hatalı: {e}");
            return 1;
        }
    };
    let hotkey = policy.hotkey.clone();
    let guard = match Guard::new(policy) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("başlatılamadı: {e}");
            return 1;
        }
    };
    let out: Out = Arc::new(Mutex::new(Box::new(std::io::stdout())));
    let server = Server::new(guard, out);
    server.spawn_ticker();

    if let Some(spec) = hotkey {
        let (mods, vk) = bekci::sys::parse_hotkey(&spec).expect("politikada doğrulandı");
        let s = server.clone();
        std::thread::spawn(move || {
            let s2 = s.clone();
            let ok = bekci::sys::hotkey_loop(mods, vk, move || {
                let mut ev = s2.stop(&format!("hotkey:{spec}"));
                ev["event"] = json!("stop");
                s2.emit(&ev);
            });
            if !ok {
                let _ = s.guard.lock().map(|mut g| g.log_event("warning", &[("detail", json!("acil durdurma kısayolu kaydedilemedi (başka program kullanıyor olabilir)"))]));
                s.emit(&json!({"event": "warning", "detail": "acil durdurma kısayolu kaydedilemedi; 'stop' işlemi çalışmaya devam eder"}));
            }
        });
    }

    let stdin = std::io::stdin();
    let mut reader = stdin.lock();
    let mut buf = Vec::new();
    loop {
        buf.clear();
        let n = match reader.by_ref().take(MAX_LINE as u64 + 1).read_until(b'\n', &mut buf) {
            Ok(n) => n,
            Err(_) => break,
        };
        if n == 0 {
            break; // istemci kapandı
        }
        if buf.len() > MAX_LINE && buf.last() != Some(&b'\n') {
            // aşırı uzun satırın kalanını at
            let mut skip = Vec::new();
            let _ = reader.read_until(b'\n', &mut skip);
            server.emit(&json!({"ok": false, "error": format!("istek satırı çok uzun (en çok {MAX_LINE} bayt)")}));
            continue;
        }
        let line = match std::str::from_utf8(&buf) {
            Ok(s) => s.trim(),
            Err(_) => {
                server.emit(&json!({"ok": false, "error": "istek UTF-8 değil"}));
                continue;
            }
        };
        if line.is_empty() {
            continue;
        }
        let (resp, quit) = server.handle_line(line);
        server.emit(&resp);
        if quit {
            break;
        }
    }
    let _ = server.guard.lock().map(|mut g| g.log_event("exit", &[]));
    let _ = std::io::stdout().flush();
    0
}
