//! Katı JSON: yinelenen anahtar hata sayılır.
//!
//! serde_json yinelenen anahtarda sessizce sonuncuyu tutar; `{"name":"not_listele","name":"hepsini_sil"}` gibi bir
//! girdi, okuyana göre farklı bir çağrı gibi görünebilir. Modelden ve istemciden gelen her şey buradan geçer.

use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number, Value};
use std::fmt;

struct Strict(Value);

impl<'de> Deserialize<'de> for Strict {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(StrictVisitor).map(Strict)
    }
}

struct StrictVisitor;

impl<'de> Visitor<'de> for StrictVisitor {
    type Value = Value;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("JSON değeri")
    }
    fn visit_bool<E>(self, v: bool) -> Result<Value, E> {
        Ok(Value::Bool(v))
    }
    fn visit_i64<E>(self, v: i64) -> Result<Value, E> {
        Ok(Value::Number(v.into()))
    }
    fn visit_u64<E>(self, v: u64) -> Result<Value, E> {
        Ok(Value::Number(v.into()))
    }
    fn visit_f64<E: de::Error>(self, v: f64) -> Result<Value, E> {
        Number::from_f64(v).map(Value::Number).ok_or_else(|| E::custom("sonlu olmayan sayı"))
    }
    fn visit_str<E>(self, v: &str) -> Result<Value, E> {
        Ok(Value::String(v.to_owned()))
    }
    fn visit_string<E>(self, v: String) -> Result<Value, E> {
        Ok(Value::String(v))
    }
    fn visit_unit<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }
    fn visit_none<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
        let mut out = Vec::new();
        while let Some(Strict(v)) = seq.next_element()? {
            out.push(v);
        }
        Ok(Value::Array(out))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        let mut out = Map::new();
        while let Some(k) = map.next_key::<String>()? {
            if out.contains_key(&k) {
                return Err(de::Error::custom(format!("yinelenen anahtar: {k}")));
            }
            let Strict(v) = map.next_value()?;
            out.insert(k, v);
        }
        Ok(Value::Object(out))
    }
}

/// Tek bir JSON değeri ayrıştırır (sonda yalnızca boşluk olabilir); yinelenen anahtarı reddeder.
/// İç içelik serde_json'un 128 düzey sınırıyla korunur.
pub fn parse_strict(text: &str) -> Result<Value, String> {
    let mut d = serde_json::Deserializer::from_str(text);
    let Strict(v) = Strict::deserialize(&mut d).map_err(|e| format!("geçersiz JSON: {e}"))?;
    d.end().map_err(|e| format!("geçersiz JSON: {e}"))?;
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_keys_rejected_at_any_depth() {
        assert!(parse_strict(r#"{"a":1,"a":2}"#).is_err());
        assert!(parse_strict(r#"{"x":{"b":1,"b":1}}"#).is_err());
        assert!(parse_strict(r#"[{"k":true,"k":false}]"#).is_err());
    }

    #[test]
    fn ordinary_json_parses() {
        let v = parse_strict(r#" {"name":"open_app","arguments":{"name":"calc","n":-3,"f":1.5}} "#).unwrap();
        assert_eq!(v["arguments"]["n"], -3);
        assert!(v["arguments"]["f"].is_f64());
    }

    #[test]
    fn trailing_garbage_and_deep_nesting_rejected() {
        assert!(parse_strict(r#"{"a":1} {"b":2}"#).is_err());
        let deep = "[".repeat(1000) + &"]".repeat(1000);
        assert!(parse_strict(&deep).is_err());
    }
}
