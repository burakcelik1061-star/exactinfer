//! bekçi: modelin araç çağrılarını doğrulayan, onaya bağlayan ve kaydeden ayrı süreç (engine-lab, Faz 4).
//! Model hiçbir şeyi doğrudan çalıştırmaz; çalıştırıcı (asistan) yalnızca bekçinin izin verdiği çağrıyı,
//! bekçinin döndürdüğü normalleştirilmiş argümanlarla çalıştırır.

pub mod calls;
pub mod guard;
pub mod jsonx;
pub mod log;
pub mod policy;
pub mod server;
pub mod sys;
pub mod validate;
