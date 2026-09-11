use crate::Error;

#[derive(Debug, Clone)]
pub struct Config {
    pub block_size: usize,
    pub num_kv_blocks: usize,
    pub max_running_requests: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            block_size: 16,
            num_kv_blocks: 128,
            max_running_requests: 8,
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<(), Error> {
        if self.block_size == 0 || self.num_kv_blocks == 0 || self.max_running_requests == 0 {
            return Err(Error::InvalidConfig("all limits must be greater than zero"));
        }
        self.block_size
            .checked_mul(self.num_kv_blocks)
            .ok_or(Error::InvalidConfig("total token capacity overflows usize"))?;
        Ok(())
    }
}
