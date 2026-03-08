use std::{fs, path::Path};

use crate::error::AppError;

pub fn detect_skills(dir: &Path) -> Result<Vec<String>, AppError> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut names = Vec::new();
    let entries = fs::read_dir(dir).map_err(|err| {
        AppError::Runtime(format!(
            "failed to read skills directory {}: {err}",
            dir.display()
        ))
    })?;
    for entry in entries {
        let entry =
            entry.map_err(|err| AppError::Runtime(format!("failed to read skill entry: {err}")))?;
        if entry.path().is_dir() {
            names.push(entry.file_name().to_string_lossy().to_string());
        }
    }
    names.sort();
    Ok(names)
}
