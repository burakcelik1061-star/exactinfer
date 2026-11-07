//! actions.log: her denetim, onay, sonuç ve acil durdurma zaman damgalı bir JSON satırı olarak yazılır.
//! Sır görünümlü alanlar (anahtar adı ya da `sk-…` değerleri) maskelenir. Yazılamazsa çağıran işlemi reddeder:
//! kaydı olmayan eylem olmaz.

use serde_json::{Map, Value};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const MAX_BYTES: u64 = 5 * 1024 * 1024;
const KEEP: usize = 3;
const SECRET_KEYS: [&str; 7] = ["api_key", "apikey", "secret", "password", "parola", "şifre", "token"];

pub struct ActionLog {
    path: PathBuf,
    file: File,
}

/// UTC zaman damgası, milisaniyeli: 2026-11-03T12:34:56.789Z
pub fn timestamp() -> String {
    let d = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let (secs, ms) = (d.as_secs() as i64, d.subsec_millis());
    let (days, sod) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    // Howard Hinnant'ın civil_from_days algoritması
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{ms:03}Z", sod / 3600, sod % 3600 / 60, sod % 60)
}

fn mask_str(s: &str) -> String {
    // sk-… biçimli API anahtarları (en az 8 karakterlik gövde)
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find("sk-") {
        let body = rest[i + 3..].chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-').count();
        out.push_str(&rest[..i]);
        if body >= 8 {
            out.push_str("***");
        } else {
            out.push_str(&rest[i..i + 3 + body]);
        }
        rest = &rest[i + 3 + body..];
    }
    out.push_str(rest);
    out
}

pub fn mask(v: &Value) -> Value {
    match v {
        Value::Object(m) => Value::Object(
            m.iter()
                .map(|(k, v)| {
                    let lk = k.to_lowercase();
                    let secret = SECRET_KEYS.iter().any(|w| lk.contains(w));
                    (k.clone(), if secret { Value::String("***".into()) } else { mask(v) })
                })
                .collect(),
        ),
        Value::Array(a) => Value::Array(a.iter().map(mask).collect()),
        Value::String(s) => Value::String(mask_str(s)),
        o => o.clone(),
    }
}

impl ActionLog {
    pub fn open(path: &Path) -> Result<ActionLog, String> {
        if let Some(d) = path.parent() {
            std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
        }
        let file = Self::open_file(path)?;
        Ok(ActionLog { path: path.to_path_buf(), file })
    }

    fn open_file(path: &Path) -> Result<File, String> {
        OpenOptions::new().create(true).append(true).open(path).map_err(|e| format!("{}: {e}", path.display()))
    }

    fn rotate_if_needed(&mut self) -> Result<(), String> {
        let size = self.file.metadata().map(|m| m.len()).unwrap_or(0);
        if size <= MAX_BYTES {
            return Ok(());
        }
        let name = |i: usize| PathBuf::from(format!("{}.{i}", self.path.display()));
        for i in (1..=KEEP).rev() {
            let src = if i == 1 { self.path.clone() } else { name(i - 1) };
            if src.exists() {
                let _ = std::fs::remove_file(name(i));
                std::fs::rename(&src, name(i)).map_err(|e| format!("günlük döndürülemedi: {e}"))?;
            }
        }
        self.file = Self::open_file(&self.path)?;
        Ok(())
    }

    /// `{"ts", "event", ...fields}` satırını yazar ve diske boşaltır.
    pub fn write(&mut self, event: &str, fields: Map<String, Value>) -> Result<(), String> {
        let mut rec = Map::new();
        rec.insert("ts".into(), Value::String(timestamp()));
        rec.insert("event".into(), Value::String(event.into()));
        for (k, v) in fields {
            rec.insert(k, v);
        }
        let line = serde_json::to_string(&mask(&Value::Object(rec))).map_err(|e| e.to_string())? + "\n";
        self.rotate_if_needed()?;
        self.file.write_all(line.as_bytes()).and_then(|_| self.file.flush()).map_err(|e| format!("actions.log yazılamadı: {e}"))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn timestamp_format() {
        let t = timestamp();
        assert_eq!(t.len(), 24);
        assert!(t.starts_with("20") && t.ends_with('Z') && &t[10..11] == "T");
    }

    #[test]
    fn masking() {
        let v = json!({"args": {"api_key": "x", "Parola": "y", "text": "anahtar sk-abcdefghijk ve sk-kısa"}, "n": 3});
        let m = mask(&v);
        assert_eq!(m["args"]["api_key"], "***");
        assert_eq!(m["args"]["Parola"], "***");
        assert_eq!(m["args"]["text"], "anahtar *** ve sk-kısa");
        assert_eq!(m["n"], 3);
    }

    #[test]
    fn writes_lines() {
        let dir = std::env::temp_dir().join(format!("bekci-log-{}", std::process::id()));
        let p = dir.join("actions.log");
        let mut l = ActionLog::open(&p).unwrap();
        let mut f = Map::new();
        f.insert("tool".into(), json!("x"));
        l.write("check", f.clone()).unwrap();
        l.write("result", f).unwrap();
        let text = std::fs::read_to_string(&p).unwrap();
        let lines: Vec<Value> = text.lines().map(|s| serde_json::from_str(s).unwrap()).collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["event"], "check");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
