//! Politika dosyası (TOML): araç beyaz listesi, parametre şemaları, izinli klasörler.
//!
//! Yükleme sırasında zorlanan kurallar:
//! - Listede olmayan araç yoktur; `blocked` araç modele hiç gösterilmez.
//! - Serbest komut (`kind = "shell"`) yalnızca `[bekci] shell_mode = true` ile yüklenir ve her zaman onay ister.
//! - Ağ kullanan araç (`network = true`) her zaman onay ister. `[bekci] network_session = true` ise kullanıcı ilk ağ
//!   onayında oturum izni de verebilir (yalnız salt okunur ağ araçları için; konuşma kirliyken askıda, durdurmada düşer).
//! - Dosyada sır tutulmaz: anahtar adı api_key/secret/password/token benzeri olan politika reddedilir.

use crate::validate;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Danger {
    Safe,
    Confirm,
    Blocked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfirmUi {
    /// Onay penceresini bekçi kendisi açar (Windows iletişim kutusu, varsayılan düğme "Hayır").
    Dialog,
    /// Onayı istemci (asistan arayüzü) toplar ve `confirm` işlemiyle bildirir.
    Client,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Kind {
    Text { max_len: usize, multiline: bool },
    Name { max_len: usize },
    Integer { min: i64, max: i64 },
    Number { min: f64, max: f64 },
    Boolean,
    Enum(Vec<String>),
    Path,
    Url,
    Shell { max_len: usize },
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub kind: Kind,
    pub required: bool,
    pub desc: String,
}

#[derive(Debug, Clone)]
pub struct Tool {
    pub name: String,
    pub danger: Danger,
    /// Yan etkisi yok (yalnızca okur). Güvenilmez veriden sonra yan etkili araçlar onaya yükseltilir.
    pub readonly: bool,
    pub network: bool,
    /// Çıktısı üçüncü taraf içeriği (dosya metni, web sayfası, pano): sonuç geldiğinde oturum "kirli" sayılır.
    pub untrusted: bool,
    pub desc: String,
    pub params: Vec<Param>,
}

#[derive(Debug, Clone)]
pub struct Policy {
    pub log: PathBuf,
    pub hotkey: Option<String>,
    pub confirm_timeout: Duration,
    pub confirm_ui: ConfirmUi,
    pub shell_mode: bool,
    pub network_session: bool,
    pub roots: Vec<PathBuf>,
    pub deny: Vec<PathBuf>,
    pub tools: Vec<Tool>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPolicy {
    #[serde(default)]
    bekci: RawBekci,
    #[serde(default)]
    paths: RawPaths,
    #[serde(default, rename = "tool")]
    tools: Vec<RawTool>,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct RawBekci {
    log: Option<String>,
    hotkey: Option<String>,
    confirm_timeout_s: Option<u64>,
    confirm_ui: Option<String>,
    #[serde(default)]
    shell_mode: bool,
    #[serde(default)]
    network_session: bool,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct RawPaths {
    #[serde(default)]
    roots: Vec<String>,
    #[serde(default)]
    deny: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTool {
    name: String,
    danger: String,
    desc: String,
    #[serde(default)]
    readonly: bool,
    #[serde(default)]
    network: bool,
    #[serde(default)]
    untrusted: bool,
    #[serde(default)]
    params: BTreeMap<String, RawParam>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawParam {
    kind: String,
    #[serde(default)]
    required: bool,
    desc: Option<String>,
    max_len: Option<usize>,
    min: Option<f64>,
    max: Option<f64>,
    values: Option<Vec<String>>,
}

const SECRET_WORDS: [&str; 7] = ["api_key", "apikey", "secret", "password", "parola", "şifre", "token"];

fn reject_secret_keys(v: &toml::Value, at: &str) -> Result<(), String> {
    match v {
        toml::Value::Table(t) => {
            for (k, v) in t {
                let lk = k.to_lowercase();
                if SECRET_WORDS.iter().any(|w| lk.contains(w)) {
                    return Err(format!("{at}{k}: sırlar (API anahtarı, parola) politika dosyasına yazılmaz"));
                }
                reject_secret_keys(v, &format!("{at}{k}."))?;
            }
        }
        toml::Value::Array(a) => {
            for (i, v) in a.iter().enumerate() {
                reject_secret_keys(v, &format!("{at}[{i}]."))?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// `%AD%` biçimindeki ortam değişkenlerini açar; tanımsız değişken hatadır (sessizce boş yola dönmesin).
pub fn expand_env(s: &str) -> Result<String, String> {
    let mut out = String::new();
    let mut rest = s;
    while let Some(i) = rest.find('%') {
        out.push_str(&rest[..i]);
        let after = &rest[i + 1..];
        let j = after.find('%').ok_or_else(|| format!("kapanmamış %: {s}"))?;
        let name = &after[..j];
        out.push_str(&std::env::var(name).map_err(|_| format!("tanımsız ortam değişkeni %{name}%"))?);
        rest = &after[j + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

fn ident_ok(s: &str) -> bool {
    let b = s.as_bytes();
    !b.is_empty() && b.len() <= 64 && b[0].is_ascii_lowercase() && b.iter().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == b'_')
}

fn dir_path(raw: &str, what: &str) -> Result<PathBuf, String> {
    let p = validate::check_path_syntax(&expand_env(raw)?).map_err(|e| format!("{what} {raw}: {e}"))?;
    validate::resolve(&p).map_err(|e| format!("{what} {raw}: {e}"))
}

fn parse_kind(name: &str, p: &RawParam, shell_mode: bool) -> Result<Kind, String> {
    let int = |v: Option<f64>, d: i64| -> Result<i64, String> {
        match v {
            None => Ok(d),
            Some(x) if x.fract() == 0.0 && x.abs() < 9.0e15 => Ok(x as i64),
            Some(_) => Err(format!("{name}: integer sınırları tamsayı olmalı")),
        }
    };
    let used = |field: &str, present: bool, allowed: bool| -> Result<(), String> {
        if present && !allowed { Err(format!("{name}: '{field}' bu türde kullanılmaz")) } else { Ok(()) }
    };
    let k = p.kind.as_str();
    used("max_len", p.max_len.is_some(), matches!(k, "text" | "multiline" | "name" | "shell"))?;
    used("min/max", p.min.is_some() || p.max.is_some(), matches!(k, "integer" | "number"))?;
    used("values", p.values.is_some(), k == "enum")?;
    Ok(match k {
        "text" => Kind::Text { max_len: p.max_len.unwrap_or(2000), multiline: false },
        "multiline" => Kind::Text { max_len: p.max_len.unwrap_or(8000), multiline: true },
        "name" => Kind::Name { max_len: p.max_len.unwrap_or(64) },
        "integer" => {
            let (min, max) = (int(p.min, i64::MIN)?, int(p.max, i64::MAX)?);
            if min > max {
                return Err(format!("{name}: min > max"));
            }
            Kind::Integer { min, max }
        }
        "number" => {
            let (min, max) = (p.min.unwrap_or(f64::MIN), p.max.unwrap_or(f64::MAX));
            if min.is_nan() || max.is_nan() || min > max {
                return Err(format!("{name}: min > max"));
            }
            Kind::Number { min, max }
        }
        "boolean" => Kind::Boolean,
        "enum" => {
            let v = p.values.clone().unwrap_or_default();
            if v.is_empty() {
                return Err(format!("{name}: enum için 'values' gerekli"));
            }
            for x in &v {
                validate::check_name(x, 128).map_err(|e| format!("{name}: enum değeri {x:?}: {e}"))?;
            }
            Kind::Enum(v)
        }
        "path" => Kind::Path,
        "url" => Kind::Url,
        "shell" => {
            if !shell_mode {
                return Err(format!(
                    "{name}: serbest komut parametresi yalnızca [bekci] shell_mode = true ile açıkça etkinleştirilebilir"
                ));
            }
            Kind::Shell { max_len: p.max_len.unwrap_or(1000) }
        }
        other => return Err(format!("{name}: bilinmeyen tür '{other}'")),
    })
}

impl Policy {
    pub fn load(path: &Path) -> Result<Policy, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut p = Policy::parse(&text)?;
        // Politika dosyasının klasörü araçlara kapalı: model kendi kurallarını yeniden yazamaz.
        let dir = path.parent().map(|d| if d.as_os_str().is_empty() { Path::new(".") } else { d });
        if let Some(d) = dir.and_then(|d| std::fs::canonicalize(d).ok()) {
            p.deny.push(validate::resolve(&d)?);
        }
        Ok(p)
    }

    pub fn parse(text: &str) -> Result<Policy, String> {
        let raw_value: toml::Value = toml::from_str(text).map_err(|e| format!("politika: {e}"))?;
        reject_secret_keys(&raw_value, "")?;
        let raw: RawPolicy = toml::from_str(text).map_err(|e| format!("politika: {e}"))?;

        let log = match &raw.bekci.log {
            Some(l) => PathBuf::from(expand_env(l)?),
            None => PathBuf::from(expand_env("%LOCALAPPDATA%")?).join("engine-lab").join("actions.log"),
        };
        let confirm_ui = match raw.bekci.confirm_ui.as_deref() {
            None | Some("dialog") => ConfirmUi::Dialog,
            Some("client") => ConfirmUi::Client,
            Some(o) => return Err(format!("confirm_ui: 'dialog' ya da 'client' olmalı, '{o}' değil")),
        };
        let timeout = raw.bekci.confirm_timeout_s.unwrap_or(60);
        if !(1..=3600).contains(&timeout) {
            return Err("confirm_timeout_s 1..3600 olmalı".into());
        }
        if let Some(h) = &raw.bekci.hotkey {
            crate::sys::parse_hotkey(h)?;
        }

        let roots = raw.paths.roots.iter().map(|r| dir_path(r, "kök")).collect::<Result<Vec<_>, _>>()?;
        let mut deny = raw.paths.deny.iter().map(|r| dir_path(r, "korumalı")).collect::<Result<Vec<_>, _>>()?;
        if let Some(d) = log.parent() {
            deny.push(validate::resolve(d)?); // actions.log'un klasörü de kapalı
        }

        let mut tools: Vec<Tool> = Vec::new();
        for t in raw.tools {
            if !ident_ok(&t.name) {
                return Err(format!("araç adı '{}': küçük harf, rakam ve _ (en çok 64)", t.name));
            }
            if tools.iter().any(|x| x.name == t.name) {
                return Err(format!("araç iki kez tanımlı: {}", t.name));
            }
            let mut danger = match t.danger.as_str() {
                "safe" => Danger::Safe,
                "confirm" => Danger::Confirm,
                "blocked" => Danger::Blocked,
                o => return Err(format!("{}: danger 'safe' | 'confirm' | 'blocked' olmalı, '{o}' değil", t.name)),
            };
            if t.network && danger == Danger::Safe {
                return Err(format!("{}: ağ kullanan araç (network = true) 'confirm' olmalı", t.name));
            }
            validate::check_text(&t.desc, 500, false).map_err(|e| format!("{}: desc {e}", t.name))?;
            let mut params = Vec::new();
            for (pn, rp) in &t.params {
                if !ident_ok(pn) {
                    return Err(format!("{}.{pn}: parametre adı küçük harf, rakam ve _ olmalı", t.name));
                }
                let kind = parse_kind(&format!("{}.{pn}", t.name), rp, raw.bekci.shell_mode)?;
                if matches!(kind, Kind::Shell { .. }) {
                    danger = if danger == Danger::Blocked { Danger::Blocked } else { Danger::Confirm };
                }
                let desc = rp.desc.clone().unwrap_or_default();
                validate::check_text(&desc, 300, false).map_err(|e| format!("{}.{pn}: desc {e}", t.name))?;
                params.push(Param { name: pn.clone(), kind, required: rp.required, desc });
            }
            if params.iter().any(|p| p.kind == Kind::Path) && roots.is_empty() {
                return Err(format!("{}: yol parametresi var ama [paths] roots boş", t.name));
            }
            let has_shell = params.iter().any(|p| matches!(p.kind, Kind::Shell { .. }));
            tools.push(Tool {
                name: t.name,
                danger,
                readonly: t.readonly && !has_shell,
                network: t.network,
                untrusted: t.untrusted,
                desc: t.desc,
                params,
            });
        }
        Ok(Policy {
            log,
            hotkey: raw.bekci.hotkey,
            confirm_timeout: Duration::from_secs(timeout),
            confirm_ui,
            shell_mode: raw.bekci.shell_mode,
            network_session: raw.bekci.network_session,
            roots,
            deny,
            tools,
        })
    }

    pub fn tool(&self, name: &str) -> Option<&Tool> {
        self.tools.iter().find(|t| t.name == name)
    }

    /// Modelin istemine konacak araç listesi (OpenAI/Qwen3 "function" biçimi); `blocked` araçlar yok.
    pub fn tools_schema(&self) -> Value {
        let list: Vec<Value> = self
            .tools
            .iter()
            .filter(|t| t.danger != Danger::Blocked)
            .map(|t| {
                let mut props = serde_json::Map::new();
                for p in &t.params {
                    // `format`: motorun kısıtlı çözümlemesi bu notla değer kuralını seçer (tek satır metin varsayılan)
                    let mut s = match &p.kind {
                        Kind::Text { max_len, multiline: false } | Kind::Shell { max_len } => {
                            json!({"type": "string", "maxLength": max_len})
                        }
                        Kind::Text { max_len, multiline: true } => {
                            json!({"type": "string", "maxLength": max_len, "format": "multiline"})
                        }
                        Kind::Name { max_len } => json!({"type": "string", "maxLength": max_len, "format": "name"}),
                        Kind::Integer { min, max } => {
                            let mut o = json!({"type": "integer"});
                            if *min != i64::MIN { o["minimum"] = json!(min); }
                            if *max != i64::MAX { o["maximum"] = json!(max); }
                            o
                        }
                        Kind::Number { min, max } => {
                            let mut o = json!({"type": "number"});
                            if *min != f64::MIN { o["minimum"] = json!(min); }
                            if *max != f64::MAX { o["maximum"] = json!(max); }
                            o
                        }
                        Kind::Boolean => json!({"type": "boolean"}),
                        Kind::Enum(v) => json!({"type": "string", "enum": v}),
                        Kind::Path => json!({"type": "string", "maxLength": 1000, "format": "path", "description": "Mutlak Windows yolu"}),
                        Kind::Url => json!({"type": "string", "maxLength": 2000, "format": "uri", "description": "https:// adresi"}),
                    };
                    if !p.desc.is_empty() {
                        s["description"] = json!(p.desc);
                    }
                    props.insert(p.name.clone(), s);
                }
                let req: Vec<&str> = t.params.iter().filter(|p| p.required).map(|p| p.name.as_str()).collect();
                let tag = if t.danger == Danger::Confirm { " [Onay gerektirir]" } else { "" };
                json!({"type": "function", "function": {
                    "name": t.name,
                    "description": format!("{}{tag}", t.desc),
                    "parameters": {"type": "object", "properties": props, "required": req, "additionalProperties": false},
                }})
            })
            .collect();
        Value::Array(list)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: &str = r#"
[bekci]
log = "%TEMP%\\bekci-test\\actions.log"
confirm_ui = "client"
[paths]
roots = ["%TEMP%"]
"#;

    #[test]
    fn loads_tools_and_schema() {
        let p = Policy::parse(&format!(
            "{BASE}
[[tool]]
name = \"open_app\"
danger = \"safe\"
desc = \"Uygulama açar\"
[tool.params.name]
kind = \"enum\"
values = [\"calc\", \"notepad\"]
required = true

[[tool]]
name = \"gizli\"
danger = \"blocked\"
desc = \"Gösterilmez\"
"
        ))
        .unwrap();
        assert_eq!(p.tools.len(), 2);
        let s = p.tools_schema();
        assert_eq!(s.as_array().unwrap().len(), 1);
        assert_eq!(s[0]["function"]["parameters"]["properties"]["name"]["enum"][1], "notepad");
        assert_eq!(s[0]["function"]["parameters"]["additionalProperties"], false);
    }

    #[test]
    fn rules_enforced_at_load() {
        let shell = format!("{BASE}\n[[tool]]\nname = \"run\"\ndanger = \"safe\"\ndesc = \"x\"\n[tool.params.cmd]\nkind = \"shell\"\n");
        assert!(Policy::parse(&shell).unwrap_err().contains("shell_mode"));
        let shell_on = shell.replace("confirm_ui = \"client\"", "confirm_ui = \"client\"\nshell_mode = true");
        let p = Policy::parse(&shell_on).unwrap();
        assert_eq!(p.tools[0].danger, Danger::Confirm, "shell her zaman onaylı");

        let net = format!("{BASE}\n[[tool]]\nname = \"web\"\ndanger = \"safe\"\nnetwork = true\ndesc = \"x\"\n");
        assert!(Policy::parse(&net).is_err());

        let secret = format!("{BASE}\napi_key = \"sk-123\"\n");
        assert!(Policy::parse(&secret).unwrap_err().contains("sır"));

        let dup = format!("{BASE}\n[[tool]]\nname = \"a\"\ndanger = \"safe\"\ndesc = \"x\"\n[[tool]]\nname = \"a\"\ndanger = \"safe\"\ndesc = \"y\"\n");
        assert!(Policy::parse(&dup).is_err());

        let unknown = format!("{BASE}\n[[tool]]\nname = \"a\"\ndanger = \"safe\"\ndesc = \"x\"\nrun_as_admin = true\n");
        assert!(Policy::parse(&unknown).is_err(), "bilinmeyen alan reddedilir");

        let badname = format!("{BASE}\n[[tool]]\nname = \"Open-App\"\ndanger = \"safe\"\ndesc = \"x\"\n");
        assert!(Policy::parse(&badname).is_err());
    }

    #[test]
    fn env_expansion() {
        std::env::set_var("BEKCI_TEST_DIR", "C:\\x");
        assert_eq!(expand_env("%BEKCI_TEST_DIR%\\y").unwrap(), "C:\\x\\y");
        assert!(expand_env("%BEKCI_YOK_BOYLE%").is_err());
        assert!(expand_env("%BEKCI_TEST_DIR").is_err());
    }
}
