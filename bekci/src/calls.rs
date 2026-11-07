//! Model çıktısından araç çağrılarını ayıklama. Qwen3 biçimi:
//!   <tool_call>
//!   {"name": "open_app", "arguments": {"name": "calc"}}
//!   </tool_call>
//! Etiket yoksa ve metnin tamamı bir JSON nesnesiyse o da çağrı sayılır; aksi halde düz yanıttır (çağrı yok).

use crate::jsonx::parse_strict;
use serde_json::Value;

pub const MAX_CALLS: usize = 8;
const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";

/// Çağrıları sırayla döndürür. Kapanmamış etiket, bozuk JSON ya da çok sayıda çağrı tüm mesajı reddettirir:
/// yarım okunmuş bir mesajdan seçme yapılmaz.
pub fn extract_calls(text: &str) -> Result<Vec<Value>, String> {
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find(OPEN) {
        let body_start = i + OPEN.len();
        let j = rest[body_start..].find(CLOSE).ok_or("kapanmamış <tool_call> etiketi")?;
        let body = &rest[body_start..body_start + j];
        if body.contains(OPEN) {
            return Err("iç içe <tool_call> etiketi".into());
        }
        calls.push(parse_strict(body.trim())?);
        if calls.len() > MAX_CALLS {
            return Err(format!("bir mesajda en çok {MAX_CALLS} araç çağrısı"));
        }
        rest = &rest[body_start + j + CLOSE.len()..];
    }
    if rest.contains(CLOSE) {
        return Err("açılmamış </tool_call> etiketi".into());
    }
    if calls.is_empty() {
        let t = text.trim();
        if t.starts_with('{') && t.ends_with('}') {
            calls.push(parse_strict(t)?);
        }
    }
    Ok(calls)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_format() {
        let t = "Hesap makinesini açıyorum.\n<tool_call>\n{\"name\": \"open_app\", \"arguments\": {\"name\": \"calc\"}}\n</tool_call>";
        let c = extract_calls(t).unwrap();
        assert_eq!(c.len(), 1);
        assert_eq!(c[0]["name"], "open_app");
    }

    #[test]
    fn bare_json_and_plain_text() {
        assert_eq!(extract_calls(r#" {"name":"x","arguments":{}} "#).unwrap().len(), 1);
        assert!(extract_calls("Ankara Türkiye'nin başkentidir.").unwrap().is_empty());
    }

    #[test]
    fn malformed_rejected() {
        assert!(extract_calls("<tool_call>{\"name\":\"x\"}").is_err());
        assert!(extract_calls("<tool_call>{\"name\":\"x\",}</tool_call>").is_err());
        assert!(extract_calls("<tool_call><tool_call>{}</tool_call></tool_call>").is_err());
        assert!(extract_calls("metin </tool_call>").is_err());
        assert!(extract_calls(r#"<tool_call>{"name":"a","name":"b","arguments":{}}</tool_call>"#).is_err());
        let many = "<tool_call>{\"name\":\"x\",\"arguments\":{}}</tool_call>".repeat(MAX_CALLS + 1);
        assert!(extract_calls(&many).is_err());
    }
}
