//! İşletim sistemi parçaları: yönetici denetimi, güvenli rastgele bilet, global kısayol, onay penceresi.

/// "ctrl+alt+end" → (değiştirici bitleri, sanal tuş kodu). Değiştiricisiz kısayol kabul edilmez.
pub fn parse_hotkey(spec: &str) -> Result<(u32, u32), String> {
    let (mut mods, mut vk) = (0u32, None);
    for part in spec.split('+').map(|p| p.trim().to_ascii_lowercase()) {
        match part.as_str() {
            "ctrl" | "control" => mods |= 0x0002,
            "alt" => mods |= 0x0001,
            "shift" => mods |= 0x0004,
            "win" => mods |= 0x0008,
            k => {
                if vk.is_some() {
                    return Err(format!("kısayolda birden çok tuş: {spec}"));
                }
                vk = Some(match k {
                    "end" => 0x23,
                    "home" => 0x24,
                    "pause" => 0x13,
                    "esc" | "escape" => 0x1B,
                    "delete" | "del" => 0x2E,
                    "insert" | "ins" => 0x2D,
                    "space" => 0x20,
                    _ if k.len() == 1 && k.as_bytes()[0].is_ascii_alphanumeric() => k.to_ascii_uppercase().as_bytes()[0] as u32,
                    _ if k.starts_with('f') && k[1..].parse::<u32>().is_ok_and(|n| (1..=24).contains(&n)) => {
                        0x6F + k[1..].parse::<u32>().unwrap()
                    }
                    _ => return Err(format!("bilinmeyen tuş '{k}' ({spec})")),
                });
            }
        }
    }
    match vk {
        Some(v) if mods != 0 => Ok((mods, v)),
        _ => Err(format!("kısayol en az bir değiştirici (ctrl/alt/shift/win) ve bir tuş içermeli: {spec}")),
    }
}

#[cfg(windows)]
mod imp {
    use std::ptr::null_mut;
    use windows_sys::Win32::Foundation::{HWND, LPARAM, WPARAM};
    use windows_sys::Win32::Security::Cryptography::{BCryptGenRandom, BCRYPT_USE_SYSTEM_PREFERRED_RNG};
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{RegisterHotKey, MOD_NOREPEAT};
    use windows_sys::Win32::UI::Shell::IsUserAnAdmin;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        FindWindowW, GetMessageW, MessageBoxW, PostMessageW, IDYES, MB_DEFBUTTON2, MB_ICONWARNING, MB_SETFOREGROUND,
        MB_SYSTEMMODAL, MB_YESNO, MSG, WM_COMMAND, WM_HOTKEY,
    };

    pub fn is_admin() -> bool {
        unsafe { IsUserAnAdmin() != 0 }
    }

    pub fn random_bytes(buf: &mut [u8]) -> Result<(), String> {
        let st = unsafe { BCryptGenRandom(null_mut(), buf.as_mut_ptr(), buf.len() as u32, BCRYPT_USE_SYSTEM_PREFERRED_RNG) };
        if st == 0 { Ok(()) } else { Err(format!("BCryptGenRandom hata {st:#x}")) }
    }

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    /// Kısayolu bu iş parçacığında kaydeder ve her basılışta `on_press`'i çağırır (sonsuza dek döner).
    /// Kayıt başarısızsa (başka program aynı kısayolu tutuyorsa) hemen `false` döner.
    pub fn hotkey_loop(mods: u32, vk: u32, on_press: impl Fn()) -> bool {
        unsafe {
            if RegisterHotKey(null_mut(), 1, mods | MOD_NOREPEAT, vk) == 0 {
                return false;
            }
            let mut msg: MSG = std::mem::zeroed();
            while GetMessageW(&mut msg, null_mut(), 0, 0) > 0 {
                if msg.message == WM_HOTKEY {
                    on_press();
                }
            }
        }
        true
    }

    /// Evet/Hayır onay penceresi; varsayılan düğme "Hayır", en üstte. Kullanıcı yanıtlayana dek bekler.
    pub fn confirm_dialog(title: &str, text: &str) -> bool {
        let (t, x) = (wide(title), wide(text));
        let r = unsafe {
            MessageBoxW(null_mut(), x.as_ptr(), t.as_ptr(),
                        MB_YESNO | MB_ICONWARNING | MB_DEFBUTTON2 | MB_SYSTEMMODAL | MB_SETFOREGROUND)
        };
        r == IDYES
    }

    /// Başlığıyla bulunan onay penceresine "Hayır" bastırır (zaman aşımı / acil durdurma).
    pub fn dismiss_dialog(title: &str) {
        let t = wide(title);
        let cls = wide("#32770");
        unsafe {
            let h: HWND = FindWindowW(cls.as_ptr(), t.as_ptr());
            if !h.is_null() {
                PostMessageW(h, WM_COMMAND, 7 as WPARAM /* IDNO */, 0 as LPARAM);
            }
        }
    }
}

#[cfg(not(windows))]
mod imp {
    pub fn is_admin() -> bool {
        false
    }
    pub fn random_bytes(buf: &mut [u8]) -> Result<(), String> {
        use std::io::Read;
        std::fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(buf)).map_err(|e| e.to_string())
    }
    pub fn hotkey_loop(_: u32, _: u32, _: impl Fn()) -> bool {
        false
    }
    pub fn confirm_dialog(_: &str, _: &str) -> bool {
        false
    }
    pub fn dismiss_dialog(_: &str) {}
}

pub use imp::{confirm_dialog, dismiss_dialog, hotkey_loop, is_admin};

/// 128 bitlik tahmin edilemez bilet (onay kimliği), onaltılık.
pub fn random_ticket() -> Result<String, String> {
    let mut b = [0u8; 16];
    imp::random_bytes(&mut b)?;
    Ok(b.iter().map(|x| format!("{x:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hotkeys() {
        assert_eq!(parse_hotkey("ctrl+alt+end").unwrap(), (0x0003, 0x23));
        assert_eq!(parse_hotkey("Ctrl + Shift + F12").unwrap(), (0x0006, 0x7B));
        assert_eq!(parse_hotkey("win+k").unwrap(), (0x0008, b'K' as u32));
        assert!(parse_hotkey("end").is_err());
        assert!(parse_hotkey("ctrl+a+b").is_err());
        assert!(parse_hotkey("ctrl+foo").is_err());
    }

    #[test]
    fn tickets_are_random() {
        let (a, b) = (random_ticket().unwrap(), random_ticket().unwrap());
        assert_eq!(a.len(), 32);
        assert_ne!(a, b);
    }
}
