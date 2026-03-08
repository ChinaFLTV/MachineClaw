use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::Mutex,
    time::{Duration, SystemTime},
};

use chrono::Local;
use once_cell::sync::OnceCell;

use crate::{
    config::{LogConfig, expand_tilde},
    error::AppError,
    mask::mask_sensitive,
};

static LOGGER: OnceCell<Mutex<LoggerState>> = OnceCell::new();

struct LoggerState {
    dir: PathBuf,
    file_name_template: String,
    max_file_size_bytes: u64,
    max_save_duration: Duration,
    session_id: String,
    sequence: u64,
    file_path: PathBuf,
    file: File,
}

pub fn init(cfg: &LogConfig, executable_dir: &Path, session_id: &str) -> Result<PathBuf, AppError> {
    let dir = resolve_log_dir(cfg, executable_dir);
    fs::create_dir_all(&dir).map_err(|err| {
        AppError::Runtime(format!(
            "failed to create log directory {}: {err}",
            dir.display()
        ))
    })?;

    let max_file_size_bytes = parse_size_to_bytes(&cfg.max_file_size)?;
    let max_save_duration = parse_retention_duration(&cfg.max_save_time)?;
    let template = cfg.log_file_name.trim().to_string();
    if template.is_empty() {
        return Err(AppError::Config(
            "log.log-file-name must include a file extension".to_string(),
        ));
    }
    if template.contains('/') || template.contains('\\') {
        return Err(AppError::Config(
            "log.log-file-name must not contain path separators".to_string(),
        ));
    }

    cleanup_old_logs(&dir, max_save_duration)?;
    let mut sequence = 1u64;
    let (file_path, file) = open_next_log_file(&dir, &template, session_id, &mut sequence)?;

    LOGGER
        .set(Mutex::new(LoggerState {
            dir,
            file_name_template: template,
            max_file_size_bytes,
            max_save_duration,
            session_id: session_id.to_string(),
            sequence,
            file_path: file_path.clone(),
            file,
        }))
        .map_err(|_| AppError::Runtime("logger already initialized".to_string()))?;

    Ok(file_path)
}

pub fn info(message: &str) {
    write_line("INFO", message);
}

pub fn warn(message: &str) {
    write_line("WARN", message);
}

pub fn error(message: &str) {
    write_line("ERROR", message);
}

fn write_line(level: &str, message: &str) {
    let sanitized = mask_sensitive(message);
    let line = format!(
        "{} [{}] {}\n",
        Local::now().format("%Y-%m-%d %H:%M:%S"),
        level,
        sanitized
    );
    if let Some(logger) = LOGGER.get()
        && let Ok(mut state) = logger.lock()
    {
        let _ = rotate_if_needed(&mut state, line.len() as u64);
        let _ = state.file.write_all(line.as_bytes());
    }
}

fn resolve_log_dir(cfg: &LogConfig, executable_dir: &Path) -> PathBuf {
    let raw = cfg.dir.trim();
    let base = if raw.is_empty() {
        PathBuf::from("logs")
    } else {
        expand_tilde(raw)
    };
    if base.is_absolute() {
        return base;
    }
    executable_dir.join(base)
}

fn parse_size_to_bytes(raw: &str) -> Result<u64, AppError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(AppError::Config(
            "log.max-file-size must not be empty".to_string(),
        ));
    }
    let split_at = trimmed
        .find(|ch: char| !ch.is_ascii_digit())
        .unwrap_or(trimmed.len());
    let (num_part, unit_part) = trimmed.split_at(split_at);
    if num_part.is_empty() {
        return Err(AppError::Config("log.max-file-size is invalid".to_string()));
    }
    let value = num_part
        .parse::<u64>()
        .map_err(|_| AppError::Config("log.max-file-size is invalid".to_string()))?;
    if value == 0 {
        return Err(AppError::Config(
            "log.max-file-size must be greater than 0".to_string(),
        ));
    }
    let unit = unit_part.trim().to_ascii_lowercase();
    let multiplier = match unit.as_str() {
        "" | "b" => 1u64,
        "kb" => 1024u64,
        "mb" => 1024u64 * 1024,
        "gb" => 1024u64 * 1024 * 1024,
        "tb" => 1024u64 * 1024 * 1024 * 1024,
        _ => return Err(AppError::Config("log.max-file-size is invalid".to_string())),
    };
    value
        .checked_mul(multiplier)
        .ok_or_else(|| AppError::Config("log.max-file-size is invalid".to_string()))
}

fn parse_retention_duration(raw: &str) -> Result<Duration, AppError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(AppError::Config(
            "log.max-save-time must not be empty".to_string(),
        ));
    }
    let split_at = trimmed
        .find(|ch: char| !ch.is_ascii_digit())
        .unwrap_or(trimmed.len());
    let (num_part, unit_part) = trimmed.split_at(split_at);
    if num_part.is_empty() {
        return Err(AppError::Config("log.max-save-time is invalid".to_string()));
    }
    let value = num_part
        .parse::<u64>()
        .map_err(|_| AppError::Config("log.max-save-time is invalid".to_string()))?;
    if value == 0 {
        return Err(AppError::Config(
            "log.max-save-time must be greater than 0".to_string(),
        ));
    }
    let seconds_per_unit = match unit_part.trim() {
        "" | "s" => 1u64,
        "m" => 60u64,
        "h" => 60u64 * 60,
        "d" => 60u64 * 60 * 24,
        "M" => 60u64 * 60 * 24 * 30,
        "y" => 60u64 * 60 * 24 * 365,
        _ => return Err(AppError::Config("log.max-save-time is invalid".to_string())),
    };
    value
        .checked_mul(seconds_per_unit)
        .map(Duration::from_secs)
        .ok_or_else(|| AppError::Config("log.max-save-time is invalid".to_string()))
}

fn rotate_if_needed(state: &mut LoggerState, incoming_bytes: u64) -> Result<(), AppError> {
    let current_size = state.file.metadata().map(|meta| meta.len()).unwrap_or(0);
    if current_size.saturating_add(incoming_bytes) <= state.max_file_size_bytes {
        return Ok(());
    }
    cleanup_old_logs(&state.dir, state.max_save_duration)?;
    let (next_path, next_file) = open_next_log_file(
        &state.dir,
        &state.file_name_template,
        &state.session_id,
        &mut state.sequence,
    )?;
    state.file = next_file;
    state.file_path = next_path;
    Ok(())
}

fn cleanup_old_logs(dir: &Path, max_age: Duration) -> Result<(), AppError> {
    let now = SystemTime::now();
    let entries = fs::read_dir(dir).map_err(|err| {
        AppError::Runtime(format!(
            "failed to read log directory {}: {err}",
            dir.display()
        ))
    })?;
    for entry in entries {
        let entry = match entry {
            Ok(value) => value,
            Err(_) => continue,
        };
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path.extension().and_then(|ext| ext.to_str()) != Some("log") {
            continue;
        }
        let modified = match entry.metadata().and_then(|meta| meta.modified()) {
            Ok(value) => value,
            Err(_) => continue,
        };
        if let Ok(elapsed) = now.duration_since(modified)
            && elapsed > max_age
        {
            let _ = fs::remove_file(path);
        }
    }
    Ok(())
}

fn open_next_log_file(
    dir: &Path,
    template: &str,
    session_id: &str,
    sequence: &mut u64,
) -> Result<(PathBuf, File), AppError> {
    for _ in 0..100_000 {
        let file_name = render_log_file_name(template, session_id, *sequence);
        let file_path = dir.join(file_name);
        *sequence += 1;
        if file_path.exists() {
            continue;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)
            .map_err(|err| {
                AppError::Runtime(format!(
                    "failed to open log file {}: {err}",
                    file_path.display()
                ))
            })?;
        return Ok((file_path, file));
    }
    Err(AppError::Runtime(
        "failed to allocate log file name after many attempts".to_string(),
    ))
}

fn render_log_file_name(template: &str, session_id: &str, sequence: u64) -> String {
    let has_seq_token = template.contains("%N");
    let with_session = template
        .replace("{session-id}", session_id)
        .replace("{session_id}", session_id);
    let with_seq_placeholder = with_session.replace("%N", "__MC_SEQ__");
    let rendered = Local::now().format(&with_seq_placeholder).to_string();
    let base = rendered.replace("__MC_SEQ__", &sequence.to_string());
    if has_seq_token {
        return base;
    }
    append_sequence_suffix(&base, sequence)
}

fn append_sequence_suffix(file_name: &str, sequence: u64) -> String {
    if sequence <= 1 {
        return file_name.to_string();
    }
    if let Some(dot_idx) = file_name.rfind('.') {
        let (name, ext) = file_name.split_at(dot_idx);
        return format!("{name}-{sequence}{ext}");
    }
    format!("{file_name}-{sequence}")
}
