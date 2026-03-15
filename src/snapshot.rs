use std::{
    fs,
    path::{Path, PathBuf},
};

use chacha20poly1305::{
    ChaCha20Poly1305, KeyInit, Nonce,
    aead::{Aead, Payload},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::{
    config::{AppConfig, parse_config_text, resolve_config_path, validate_config},
    error::AppError,
};

const SNAPSHOT_MAGIC: &[u8] = b"MACHINECLAW_SNAPSHOT_V1";
const SNAPSHOT_AAD: &[u8] = b"machineclaw.snapshot.payload.v1";
const SNAPSHOT_KDF_CONTEXT: &[u8] = b"MachineClaw::ConfigSnapshot::KeyV1";

#[derive(Debug, Clone)]
pub enum EffectiveConfigSource {
    File(PathBuf),
    Embedded(PathBuf),
    DefaultPath(PathBuf),
}

impl EffectiveConfigSource {
    pub fn describe(&self) -> String {
        match self {
            Self::File(path) => format!("file:{}", path.display()),
            Self::Embedded(path) => format!("embedded:{}", path.display()),
            Self::DefaultPath(path) => format!("default:{}", path.display()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EffectiveConfig {
    pub cfg: AppConfig,
    pub source: EffectiveConfigSource,
}

#[derive(Debug, Clone)]
pub struct SnapshotBuildResult {
    pub output_path: PathBuf,
    pub source_desc: String,
    pub bytes_written: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SnapshotMeta {
    version: u8,
    cipher: String,
    salt: [u8; 16],
    nonce: [u8; 12],
    ciphertext_len: u64,
}

#[derive(Debug)]
struct ParsedSnapshot<'a> {
    base_end: usize,
    ciphertext: &'a [u8],
    meta: SnapshotMeta,
}

pub fn load_effective_config(conf: Option<PathBuf>) -> Result<EffectiveConfig, AppError> {
    if let Some(path) = conf {
        let resolved = resolve_config_path(Some(path))?;
        let cfg = crate::config::load_config(&resolved)?;
        validate_config(&cfg)?;
        return Ok(EffectiveConfig {
            cfg,
            source: EffectiveConfigSource::File(resolved),
        });
    }

    let exe = std::env::current_exe()
        .map_err(|err| AppError::Runtime(format!("cannot locate current executable: {err}")))?;
    if let Some(raw) = try_read_embedded_snapshot(&exe)? {
        let cfg = parse_config_text(&raw, "embedded config snapshot")?;
        validate_config(&cfg)?;
        return Ok(EffectiveConfig {
            cfg,
            source: EffectiveConfigSource::Embedded(exe),
        });
    }

    let default_path = resolve_config_path(None)?;
    let cfg = crate::config::load_config(&default_path)?;
    validate_config(&cfg)?;
    Ok(EffectiveConfig {
        cfg,
        source: EffectiveConfigSource::DefaultPath(default_path),
    })
}

pub fn build_snapshot_binary(
    cfg: &AppConfig,
    source_desc: String,
    output: Option<PathBuf>,
) -> Result<SnapshotBuildResult, AppError> {
    validate_config(cfg)?;
    let config_raw = toml::to_string_pretty(cfg)
        .map_err(|err| AppError::Runtime(format!("failed to serialize config snapshot: {err}")))?;

    let exe = std::env::current_exe()
        .map_err(|err| AppError::Runtime(format!("cannot locate current executable: {err}")))?;
    let exe_bytes = fs::read(&exe).map_err(|err| {
        AppError::Runtime(format!(
            "failed to read executable {}: {err}",
            exe.display()
        ))
    })?;
    let base_bytes = match parse_snapshot(&exe_bytes)? {
        Some(parsed) => exe_bytes[..parsed.base_end].to_vec(),
        None => exe_bytes,
    };

    let (ciphertext, meta) = encrypt_snapshot(&base_bytes, config_raw.as_bytes())?;
    let meta_raw = serde_json::to_vec(&meta)
        .map_err(|err| AppError::Runtime(format!("failed to encode snapshot metadata: {err}")))?;
    if meta_raw.len() > u32::MAX as usize {
        return Err(AppError::Runtime(
            "snapshot metadata too large to append".to_string(),
        ));
    }

    let output_path = resolve_output_path(&exe, output)?;
    if output_path == exe {
        return Err(AppError::Runtime(
            "snapshot output path cannot be the current executable itself".to_string(),
        ));
    }
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            AppError::Runtime(format!(
                "failed to create snapshot output directory {}: {err}",
                parent.display()
            ))
        })?;
    }

    let mut out = Vec::with_capacity(
        base_bytes.len() + ciphertext.len() + meta_raw.len() + 4 + SNAPSHOT_MAGIC.len(),
    );
    out.extend_from_slice(&base_bytes);
    out.extend_from_slice(&ciphertext);
    out.extend_from_slice(&meta_raw);
    out.extend_from_slice(&(meta_raw.len() as u32).to_le_bytes());
    out.extend_from_slice(SNAPSHOT_MAGIC);

    fs::write(&output_path, &out).map_err(|err| {
        AppError::Runtime(format!(
            "failed to write snapshot executable {}: {err}",
            output_path.display()
        ))
    })?;
    ensure_executable_permission(&output_path)?;

    Ok(SnapshotBuildResult {
        output_path,
        source_desc,
        bytes_written: out.len(),
    })
}

pub fn render_snapshot_result(result: &SnapshotBuildResult) -> String {
    format!(
        "# Snapshot Build\n\n- status: success\n- source: {}\n- output: {}\n- size: {}\n\n说明：该二进制已内嵌配置快照，运行时可直接省略 `--conf`；如显式传入 `--conf`，会覆盖内嵌快照配置。",
        result.source_desc,
        result.output_path.display(),
        crate::i18n::human_bytes(result.bytes_written as u128)
    )
}

pub fn render_show_config(cfg: &AppConfig, source_desc: &str) -> Result<String, AppError> {
    let sanitized = sanitize_config_for_display(cfg);
    let raw = toml::to_string_pretty(&sanitized).map_err(|err| {
        AppError::Runtime(format!("failed to serialize masked config snapshot: {err}"))
    })?;
    Ok(format!(
        "# Config Snapshot\n\n- source: {source_desc}\n- sensitive_fields: masked\n\n```toml\n{}\n```",
        raw.trim_end()
    ))
}

fn resolve_output_path(exe: &Path, output: Option<PathBuf>) -> Result<PathBuf, AppError> {
    if let Some(path) = output {
        if path.is_absolute() {
            return Ok(path);
        }
        let cwd = std::env::current_dir().map_err(|err| {
            AppError::Runtime(format!("failed to resolve current directory: {err}"))
        })?;
        return Ok(cwd.join(path));
    }

    let parent = exe.parent().ok_or_else(|| {
        AppError::Runtime("failed to resolve executable directory for snapshot output".to_string())
    })?;
    let stem = exe
        .file_stem()
        .and_then(|item| item.to_str())
        .unwrap_or("MachineClaw");
    let ext = exe.extension().and_then(|item| item.to_str());
    let filename = if let Some(ext) = ext {
        format!("{stem}-snapshot.{ext}")
    } else {
        format!("{stem}-snapshot")
    };
    Ok(parent.join(filename))
}

fn encrypt_snapshot(base_bytes: &[u8], plain: &[u8]) -> Result<(Vec<u8>, SnapshotMeta), AppError> {
    let salt = Uuid::new_v4().into_bytes();
    let nonce_seed = Uuid::new_v4().into_bytes();
    let mut nonce_bytes = [0u8; 12];
    nonce_bytes.copy_from_slice(&nonce_seed[..12]);
    let key = derive_key(base_bytes, &salt);
    let cipher = ChaCha20Poly1305::new((&key).into());
    let ciphertext = cipher
        .encrypt(
            Nonce::from_slice(&nonce_bytes),
            Payload {
                msg: plain,
                aad: SNAPSHOT_AAD,
            },
        )
        .map_err(|err| AppError::Runtime(format!("failed to encrypt config snapshot: {err}")))?;
    Ok((
        ciphertext,
        SnapshotMeta {
            version: 1,
            cipher: "chacha20poly1305".to_string(),
            salt,
            nonce: nonce_bytes,
            ciphertext_len: 0, // set below
        },
    ))
    .map(|(ciphertext, mut meta)| {
        meta.ciphertext_len = ciphertext.len() as u64;
        (ciphertext, meta)
    })
}

fn try_read_embedded_snapshot(exe: &Path) -> Result<Option<String>, AppError> {
    let bytes = fs::read(exe).map_err(|err| {
        AppError::Runtime(format!(
            "failed to read executable {}: {err}",
            exe.display()
        ))
    })?;
    let Some(parsed) = parse_snapshot(&bytes)? else {
        return Ok(None);
    };
    if parsed.meta.version != 1 || parsed.meta.cipher != "chacha20poly1305" {
        return Err(AppError::Config(
            "embedded config snapshot format is not supported".to_string(),
        ));
    }
    let key = derive_key(&bytes[..parsed.base_end], &parsed.meta.salt);
    let cipher = ChaCha20Poly1305::new((&key).into());
    let plain = cipher
        .decrypt(
            Nonce::from_slice(&parsed.meta.nonce),
            Payload {
                msg: parsed.ciphertext,
                aad: SNAPSHOT_AAD,
            },
        )
        .map_err(|err| {
            AppError::Config(format!("failed to decrypt embedded config snapshot: {err}"))
        })?;
    let text = String::from_utf8(plain)
        .map_err(|err| AppError::Config(format!("embedded config snapshot is not utf-8: {err}")))?;
    Ok(Some(text))
}

fn parse_snapshot(bytes: &[u8]) -> Result<Option<ParsedSnapshot<'_>>, AppError> {
    let min_len = SNAPSHOT_MAGIC.len() + 4;
    if bytes.len() < min_len {
        return Ok(None);
    }
    if &bytes[bytes.len() - SNAPSHOT_MAGIC.len()..] != SNAPSHOT_MAGIC {
        return Ok(None);
    }

    let meta_len_pos = bytes.len() - SNAPSHOT_MAGIC.len() - 4;
    let meta_len = u32::from_le_bytes([
        bytes[meta_len_pos],
        bytes[meta_len_pos + 1],
        bytes[meta_len_pos + 2],
        bytes[meta_len_pos + 3],
    ]) as usize;
    if meta_len == 0 || meta_len > meta_len_pos {
        return Err(AppError::Config(
            "embedded config snapshot metadata is corrupted".to_string(),
        ));
    }
    let meta_start = meta_len_pos - meta_len;
    let meta: SnapshotMeta = serde_json::from_slice(&bytes[meta_start..meta_len_pos])
        .map_err(|err| AppError::Config(format!("invalid embedded config metadata: {err}")))?;
    let cipher_len = meta.ciphertext_len as usize;
    if cipher_len == 0 || cipher_len > meta_start {
        return Err(AppError::Config(
            "embedded config snapshot ciphertext is corrupted".to_string(),
        ));
    }
    let cipher_start = meta_start - cipher_len;
    Ok(Some(ParsedSnapshot {
        base_end: cipher_start,
        ciphertext: &bytes[cipher_start..meta_start],
        meta,
    }))
}

fn derive_key(base_bytes: &[u8], salt: &[u8; 16]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(base_bytes);
    hasher.update(SNAPSHOT_KDF_CONTEXT);
    hasher.update(salt);
    let digest = hasher.finalize();
    let mut key = [0u8; 32];
    key.copy_from_slice(&digest[..32]);
    key
}

fn sanitize_config_for_display(cfg: &AppConfig) -> AppConfig {
    let mut safe = cfg.clone();
    safe.ai.token = sanitize_secret_value(&safe.ai.token);
    safe
}

fn sanitize_secret_value(raw: &str) -> String {
    if raw.trim().is_empty() {
        return "".to_string();
    }
    if raw.trim_start().starts_with("sk-") {
        return "sk-****".to_string();
    }
    "****".to_string()
}

fn ensure_executable_permission(path: &Path) -> Result<(), AppError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perm = fs::metadata(path)
            .map_err(|err| {
                AppError::Runtime(format!(
                    "failed to read permissions {}: {err}",
                    path.display()
                ))
            })?
            .permissions();
        let mode = perm.mode();
        if mode & 0o111 == 0 {
            perm.set_mode(mode | 0o755);
            fs::set_permissions(path, perm).map_err(|err| {
                AppError::Runtime(format!(
                    "failed to set executable permission {}: {err}",
                    path.display()
                ))
            })?;
        }
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}
