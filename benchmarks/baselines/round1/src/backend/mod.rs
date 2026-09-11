mod mock;
pub use mock::MockBackend;

use crate::{Error, RequestId};

/// Minimal synchronous boundary; a future GPU backend will need batched prefill,
/// decode, physical KV storage and lifecycle hooks beyond this scaffold.
pub trait Backend {
    fn next_token(&mut self, request_id: RequestId, tokens: &[u32]) -> Result<u32, Error>;
}
