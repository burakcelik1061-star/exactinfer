//! Modelin ürettiği tek tek değerlerin doğrulanması: metin, yol (izinli köklere hapis), https adresi.
//! Her işlev ya normalleştirilmiş değeri ya da kullanıcıya gösterilebilir bir ret nedenini döndürür.

use std::path::{Component, Path, PathBuf, Prefix};

/// Görünmez / yön değiştiren karakterler: kontrol (Cc), biçim (Cf: sıfır genişlik, RLO…), satır ve paragraf
/// ayırıcıları. Onay penceresinde bir değerin sahte satır açmasını ya da uzantısını ters göstermesini engeller.
pub fn is_hidden(c: char) -> bool {
    c.is_control()
        || matches!(c as u32,
            0x00AD | 0x061C | 0x180E | 0x200B..=0x200F | 0x2028..=0x202E | 0x2060..=0x2064
            | 0x2066..=0x206F | 0xFEFF | 0xFFF9..=0xFFFB | 0xE0000..=0xE007F)
}

/// Serbest metin: uzunluk sınırı, NUL yok, görünmez karakter yok (`multiline` ise yalnız \n ve \t serbest).
pub fn check_text(v: &str, max_len: usize, multiline: bool) -> Result<String, String> {
    if v.chars().count() > max_len {
        return Err(format!("çok uzun (en çok {max_len} karakter)"));
    }
    if let Some(c) = v.chars().find(|&c| is_hidden(c) && !(multiline && (c == '\n' || c == '\t'))) {
        return Err(format!("görünmez/kontrol karakteri içeriyor (U+{:04X})", c as u32));
    }
    Ok(v.to_owned())
}

/// Ad benzeri değer (uygulama adı, anahtar): harf, rakam, boşluk, `_ - .`; baş ve sonda boşluk yok.
pub fn check_name(v: &str, max_len: usize) -> Result<String, String> {
    let s = check_text(v, max_len, false)?;
    if s.is_empty() || s.trim() != s {
        return Err("boş ya da başında/sonunda boşluk var".into());
    }
    if let Some(c) = s.chars().find(|&c| !(c.is_alphanumeric() || matches!(c, ' ' | '_' | '-' | '.'))) {
        return Err(format!("izin verilmeyen karakter: {c:?}"));
    }
    Ok(s)
}

/// Tarayıcıda açılacak adres: yalnız https, ana makine var, kullanıcı:parola yok, boşluk/görünmez karakter yok.
pub fn check_url(v: &str) -> Result<String, String> {
    let s = check_text(v, 2000, false)?;
    if s.chars().any(char::is_whitespace) {
        return Err("adreste boşluk olamaz".into());
    }
    let rest = s.strip_prefix("https://").ok_or("yalnızca https:// adresleri")?;
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    if authority.contains('@') {
        return Err("adreste kullanıcı adı/parola olamaz".into());
    }
    let host = authority.rsplit_once(':').map_or(authority, |(h, port)| {
        if port.chars().all(|c| c.is_ascii_digit()) { h } else { authority }
    });
    if host.is_empty() || !host.chars().all(|c| c.is_alphanumeric() || matches!(c, '.' | '-' | '[' | ']' | ':')) {
        return Err("geçersiz ana makine adı".into());
    }
    Ok(s)
}

/// Onay özetine gömülen GÜVENİLMEZ değer: satır sonu dahil tüm görünmez karakterler `\u{…}` olarak görünür olur.
pub fn display(v: &str) -> String {
    let mut out = String::with_capacity(v.len());
    for c in v.chars() {
        if is_hidden(c) {
            out.push_str(&format!("\\u{{{:04x}}}", c as u32));
        } else {
            out.push(c);
        }
    }
    out
}

fn is_device_name(part: &str) -> bool {
    let stem = part.split('.').next().unwrap_or("").to_ascii_lowercase();
    let stem = stem.trim_end();
    matches!(stem, "con" | "prn" | "aux" | "nul" | "conin$" | "conout$" | "clock$")
        || ((stem.starts_with("com") || stem.starts_with("lpt")) && stem.len() == 4 && stem.as_bytes()[3].is_ascii_digit())
        || ((stem.starts_with("com") || stem.starts_with("lpt")) && matches!(&stem[3..], "¹" | "²" | "³"))
}

/// Yolun sözdizimi denetimi (dosya sistemine dokunmadan): mutlak sürücü yolu, UNC/aygıt ad alanı yok,
/// alternatif veri akışı yok, `.`/`..` yok, aygıt adı yok, joker yok, sonda nokta/boşluk yok.
pub fn check_path_syntax(raw: &str) -> Result<PathBuf, String> {
    let s = check_text(raw, 1000, false)?;
    let s = s.trim();
    if s.is_empty() {
        return Err("boş yol".into());
    }
    if s.starts_with("\\\\") || s.starts_with("//") || s.starts_with("\\/") || s.starts_with("/\\") {
        return Err("ağ (UNC) ve aygıt yolları yasak".into());
    }
    let b = s.as_bytes();
    if b.len() < 3 || !b[0].is_ascii_alphabetic() || b[1] != b':' || !(b[2] == b'\\' || b[2] == b'/') {
        return Err("yol, sürücü harfiyle başlayan mutlak yol olmalı (ör. C:\\Users\\...)".into());
    }
    if s[2..].contains(':') {
        return Err("yolda ':' olamaz (alternatif veri akışı)".into());
    }
    for part in s[3..].split(['\\', '/']).filter(|p| !p.is_empty()) {
        if part == "." || part == ".." {
            return Err("'.' ya da '..' içeren yollar yasak".into());
        }
        if is_device_name(part) {
            return Err(format!("aygıt adı yasak: {part}"));
        }
        if part.chars().any(|c| matches!(c, '*' | '?' | '"' | '<' | '>' | '|')) {
            return Err(format!("yolda joker/özel karakter: {part}"));
        }
        if part.ends_with('.') || part.ends_with(' ') {
            return Err(format!("sonu nokta ya da boşlukla biten bileşen (Windows kırpar, başka dosyaya denk gelir): {part}"));
        }
    }
    Ok(PathBuf::from(s.replace('/', "\\")))
}

/// Var olan en uzun atayı gerçek yoluna çevirir (bağlantı noktaları, sembolik bağlar, 8.3 kısa adlar çözülür) ve
/// var olmayan kalanı ekler. Gerçek yol ağ paylaşımına çıkıyorsa reddedilir.
pub fn resolve(p: &Path) -> Result<PathBuf, String> {
    let mut existing = p.to_path_buf();
    let mut rest = Vec::new();
    loop {
        if let Ok(c) = std::fs::canonicalize(&existing) {
            let mut out = strip_verbatim(&c)?;
            for r in rest.iter().rev() {
                out.push(r);
            }
            return Ok(out);
        }
        match (existing.parent(), existing.file_name()) {
            (Some(parent), Some(name)) => {
                rest.push(name.to_owned());
                existing = parent.to_path_buf();
            }
            _ => return Ok(p.to_path_buf()),
        }
    }
}

fn strip_verbatim(p: &Path) -> Result<PathBuf, String> {
    let mut comps = p.components();
    match comps.next() {
        Some(Component::Prefix(pre)) => match pre.kind() {
            Prefix::VerbatimDisk(d) | Prefix::Disk(d) => {
                let mut out = PathBuf::from(format!("{}:\\", d as char));
                for c in comps {
                    if let Component::Normal(n) = c {
                        out.push(n);
                    }
                }
                Ok(out)
            }
            _ => Err("yol bir ağ paylaşımına ya da aygıta çözülüyor".into()),
        },
        _ => Ok(p.to_path_buf()),
    }
}

fn lower_parts(p: &Path) -> Vec<String> {
    p.components()
        .filter_map(|c| match c {
            Component::Prefix(pre) => Some(pre.as_os_str().to_string_lossy().to_lowercase()),
            Component::Normal(n) => Some(n.to_string_lossy().to_lowercase()),
            _ => None,
        })
        .collect()
}

/// `p`, `base`'in kendisi ya da altında mı (Windows: büyük/küçük harf duyarsız, bileşen bazında).
pub fn is_within(p: &Path, base: &Path) -> bool {
    let (a, b) = (lower_parts(p), lower_parts(base));
    a.len() >= b.len() && a[..b.len()] == b[..]
}

/// Yol hapsi: sözdizimi + gerçek yol + korumalı klasör + izinli kök denetimi. Normalleştirilmiş yolu döndürür.
pub fn check_path(raw: &str, roots: &[PathBuf], deny: &[PathBuf]) -> Result<String, String> {
    let p = resolve(&check_path_syntax(raw)?)?;
    if deny.iter().any(|d| is_within(&p, d)) {
        return Err("korumalı klasör (bekçinin kendi verisi/politikası) araçlara kapalı".into());
    }
    if !roots.iter().any(|r| is_within(&p, r)) {
        return Err("yol izinli klasörlerin dışında".into());
    }
    Ok(p.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidden_characters() {
        assert!(check_text("abc\u{202E}gpj.exe", 100, false).is_err());
        assert!(check_text("satır\nikinci", 100, false).is_err());
        assert!(check_text("satır\nikinci", 100, true).is_ok());
        assert!(check_text("sıfır\u{200B}genişlik", 100, true).is_err());
        assert!(check_text(&"a".repeat(11), 10, false).is_err());
        assert_eq!(display("a\nb\u{202E}"), "a\\u{000a}b\\u{202e}");
    }

    #[test]
    fn names() {
        assert!(check_name("Not Defteri", 64).is_ok());
        assert!(check_name("calc; rm", 64).is_err());
        assert!(check_name(" calc", 64).is_err());
        assert!(check_name("a&b", 64).is_err());
    }

    #[test]
    fn urls() {
        assert!(check_url("https://tr.wikipedia.org/wiki/Ankara").is_ok());
        assert!(check_url("https://example.com:8443/a?b=c").is_ok());
        assert!(check_url("http://example.com").is_err());
        assert!(check_url("file:///C:/Windows").is_err());
        assert!(check_url("https://user:pw@example.com").is_err());
        assert!(check_url("https://evil.com@good.com/").is_err());
        assert!(check_url("https://a.com/ b").is_err());
        assert!(check_url("https:///path").is_err());
        assert!(check_url("javascript:alert(1)").is_err());
    }

    #[test]
    fn path_syntax() {
        for bad in [
            "\\\\server\\share\\x", "//server/share", "\\\\?\\C:\\x", "\\\\.\\PhysicalDrive0", "C:x.txt", "x.txt",
            "\\Windows", "C:\\a\\..\\b", "C:\\a\\.\\b", "C:\\a\\file.txt:secret", "C:\\a\\CON", "C:\\a\\nul.txt",
            "C:\\a\\com1", "C:\\a\\b*", "C:\\a\\gizli.txt.", "C:\\a\\klasör \\x", "C:\\a\u{202E}txt.exe",
        ] {
            assert!(check_path_syntax(bad).is_err(), "kabul edilmemeliydi: {bad:?}");
        }
        for good in ["C:\\Users\\a\\Belgeler\\not.txt", "D:/veri/rapor 2026.pdf", "C:\\console\\com10\\lpt"] {
            assert!(check_path_syntax(good).is_ok(), "kabul edilmeliydi: {good:?}");
        }
    }

    #[test]
    fn within_is_component_wise_and_case_insensitive() {
        let base = Path::new("C:\\Users\\a\\Documents");
        assert!(is_within(Path::new("c:\\users\\A\\documents\\x.txt"), base));
        assert!(is_within(Path::new("C:\\Users\\a\\Documents"), base));
        assert!(!is_within(Path::new("C:\\Users\\a\\DocumentsSecret\\x"), base));
        assert!(!is_within(Path::new("C:\\Users"), base));
    }

    #[test]
    fn jail() {
        let tmp = std::env::temp_dir().join(format!("bekci-jail-{}", std::process::id()));
        let root = tmp.join("izinli");
        let deny = root.join("korumali");
        std::fs::create_dir_all(&deny).unwrap();
        let roots = [resolve(&root).unwrap()];
        let denies = [resolve(&deny).unwrap()];
        let ok = check_path(&format!("{}\\yeni\\not.txt", root.display()), &roots, &denies).unwrap();
        assert!(ok.to_lowercase().ends_with("izinli\\yeni\\not.txt"));
        assert!(check_path(&format!("{}\\x", deny.display()), &roots, &denies).is_err());
        assert!(check_path(&format!("{}\\x", tmp.display()), &roots, &denies).is_err());
        assert!(check_path("C:\\Windows\\System32\\drivers\\etc\\hosts", &roots, &denies).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
