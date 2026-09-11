use crate::{Backend, Error, RequestId};

/// Deterministic fake output for scheduler development; performs no inference.
#[derive(Debug, Default)]
pub struct MockBackend;

impl Backend for MockBackend {
    fn next_token(&mut self, _request_id: RequestId, tokens: &[u32]) -> Result<u32, Error> {
        tokens
            .last()
            .map(|token| token.wrapping_add(1))
            .ok_or(Error::InvalidRequest("prompt must not be empty"))
    }
}
