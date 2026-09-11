//! Single-GPU Qwen3 inference and a separately retained CPU scheduler demo.
pub mod backend;
pub mod config;
pub mod engine;
#[cfg(feature = "cuda")]
pub mod gpu;
#[cfg(feature = "cuda")]
pub mod model;
pub mod runtime;
pub mod sequence;

pub use backend::{Backend, MockBackend};
pub use config::Config;
pub use engine::Engine;
pub use sequence::{GenerationEvent, RequestId, SamplingParams};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidConfig(&'static str),
    InvalidRequest(&'static str),
    CapacityExceeded { required: usize, available: usize },
    RequestIdExhausted,
    Backend(String),
    EngineFailed,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid configuration: {message}"),
            Self::InvalidRequest(message) => write!(f, "invalid request: {message}"),
            Self::CapacityExceeded {
                required,
                available,
            } => {
                write!(
                    f,
                    "request needs {required} blocks; capacity is {available}"
                )
            }
            Self::RequestIdExhausted => write!(f, "request IDs exhausted"),
            Self::Backend(message) => write!(f, "backend failed: {message}"),
            Self::EngineFailed => write!(f, "engine terminated after a backend failure"),
        }
    }
}

impl std::error::Error for Error {}
