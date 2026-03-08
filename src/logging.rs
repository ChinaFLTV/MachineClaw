use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::Mutex,
};

use chrono::Local;
use once_cell::sync::OnceCell;

use crate::{error::AppError, mask::mask_sensitive};

static LOGGER: OnceCell<Mutex<File>> = OnceCell::new();

pub fn init(log_dir: &Path) -> Result<PathBuf, AppError> {
    fs::create_dir_all(log_dir).map_err(|err| {
        AppError::Runtime(format!(
            "failed to create log directory {}: {err}",
            log_dir.display()
        ))
    })?;
    let file_name = format!("machineclaw-{}.log", Local::now().format("%Y%m%d-%H%M%S"));
    let file_path = log_dir.join(file_name);
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
    LOGGER
        .set(Mutex::new(file))
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
    if let Some(file) = LOGGER.get()
        && let Ok(mut file) = file.lock()
    {
        let _ = file.write_all(line.as_bytes());
    }
}
