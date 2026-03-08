use thiserror::Error;

#[derive(Debug, Clone, Copy)]
pub enum ExitCode {
    Success = 0,
    GeneralFailure = 1,
    PermissionDenied = 2,
    ConfigError = 3,
    AiFailure = 4,
    CommandFailure = 5,
}

#[derive(Debug, Error)]
pub enum AppError {
    #[error("configuration error: {0}")]
    Config(String),
    #[error("permission error: {0}")]
    Permission(String),
    #[error("ai error: {0}")]
    Ai(String),
    #[error("command error: {0}")]
    Command(String),
    #[error("runtime error: {0}")]
    Runtime(String),
}

impl AppError {
    pub fn exit_code(&self) -> ExitCode {
        match self {
            Self::Config(_) => ExitCode::ConfigError,
            Self::Permission(_) => ExitCode::PermissionDenied,
            Self::Ai(_) => ExitCode::AiFailure,
            Self::Command(_) => ExitCode::CommandFailure,
            Self::Runtime(_) => ExitCode::GeneralFailure,
        }
    }
}
