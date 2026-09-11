pub type RequestId = u64;

#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    /// Fixed output length in this scaffold; EOS and sampling are not implemented.
    pub max_tokens: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self { max_tokens: 16 }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationEvent {
    pub request_id: RequestId,
    pub token_id: u32,
    pub finished: bool,
}

pub(crate) struct Sequence {
    pub id: RequestId,
    pub tokens: Vec<u32>,
    pub max_tokens: usize,
    pub generated: usize,
    pub blocks_needed: usize,
    pub blocks: Vec<usize>,
}
