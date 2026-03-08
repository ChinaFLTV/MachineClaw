use once_cell::sync::Lazy;
use regex::Regex;

static KEY_VALUE_SECRET_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(authorization|token|api[_-]?key|password|passwd|cookie|private[_-]?key)\s*[:=]\s*([^\s,;]+)")
        .expect("valid secret regex")
});

static BEARER_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)bearer\s+[A-Za-z0-9._\-+/=]{8,}").expect("valid bearer regex"));

static SK_TOKEN_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"sk-[A-Za-z0-9]{8,}").expect("valid sk token regex"));

static PRIVATE_KEY_PATH_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)([^\s]+(?:private|secret|credential)[^\s]*\.(?:pem|key|p12|pfx))")
        .expect("valid key path regex")
});

pub fn mask_sensitive(input: &str) -> String {
    let mut out = input.to_string();
    out = KEY_VALUE_SECRET_RE
        .replace_all(&out, "$1=<redacted>")
        .to_string();
    out = BEARER_RE.replace_all(&out, "Bearer <redacted>").to_string();
    out = SK_TOKEN_RE.replace_all(&out, "sk-<redacted>").to_string();
    PRIVATE_KEY_PATH_RE
        .replace_all(&out, "<redacted-key-path>")
        .to_string()
}
