use std::process::Command;

use crate::error::AppError;

#[derive(Debug, Clone, Copy)]
pub enum OsType {
    Windows,
    MacOS,
    Linux,
    Other,
}

pub fn current_os() -> OsType {
    match std::env::consts::OS {
        "windows" => OsType::Windows,
        "macos" => OsType::MacOS,
        "linux" => OsType::Linux,
        _ => OsType::Other,
    }
}

pub fn require_elevated_permissions() -> Result<(), AppError> {
    let is_elevated = match current_os() {
        OsType::Windows => check_windows_admin(),
        OsType::MacOS | OsType::Linux | OsType::Other => check_unix_root(),
    }?;

    if is_elevated {
        return Ok(());
    }

    match current_os() {
        OsType::Windows => Err(AppError::Permission(
            "administrator privileges are required on Windows".to_string(),
        )),
        _ => Err(AppError::Permission(
            "root privileges are required on Linux/macOS".to_string(),
        )),
    }
}

pub fn os_name() -> &'static str {
    match current_os() {
        OsType::Windows => "windows",
        OsType::MacOS => "macos",
        OsType::Linux => "linux",
        OsType::Other => "other",
    }
}

fn check_unix_root() -> Result<bool, AppError> {
    #[cfg(unix)]
    {
        let output = Command::new("id")
            .arg("-u")
            .output()
            .map_err(|err| AppError::Permission(format!("failed to run id -u: {err}")))?;
        let uid = String::from_utf8_lossy(&output.stdout);
        return Ok(uid.trim() == "0");
    }

    #[cfg(not(unix))]
    {
        Ok(false)
    }
}

fn check_windows_admin() -> Result<bool, AppError> {
    #[cfg(windows)]
    {
        let status = Command::new("cmd")
            .args(["/C", "net session >NUL 2>&1"])
            .status()
            .map_err(|err| {
                AppError::Permission(format!("failed to check admin permission: {err}"))
            })?;
        return Ok(status.success());
    }

    #[cfg(not(windows))]
    {
        Ok(false)
    }
}
