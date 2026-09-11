mod block_manager;
mod scheduler;

use crate::{
    Backend, Config, Error, GenerationEvent, RequestId, SamplingParams, sequence::Sequence,
};
use block_manager::BlockManager;
use scheduler::Scheduler;

/// FIFO scheduling with one fake/delegated token per running request per step.
/// Blocks cover prompt + maximum output up front and represent logical capacity
/// only. This is not actual paged KV storage, prefix caching or GPU inference.
pub struct Engine<B: Backend> {
    config: Config,
    backend: B,
    scheduler: Scheduler,
    blocks: BlockManager,
    next_id: RequestId,
    failed: bool,
}

impl<B: Backend> Engine<B> {
    pub fn new(config: Config, backend: B) -> Result<Self, Error> {
        config.validate()?;
        let blocks = BlockManager::new(config.num_kv_blocks);
        Ok(Self {
            config,
            backend,
            scheduler: Scheduler::default(),
            blocks,
            next_id: 1,
            failed: false,
        })
    }

    pub fn add_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        params: SamplingParams,
    ) -> Result<RequestId, Error> {
        if self.failed {
            return Err(Error::EngineFailed);
        }
        if prompt_tokens.is_empty() {
            return Err(Error::InvalidRequest("prompt must not be empty"));
        }
        if params.max_tokens == 0 {
            return Err(Error::InvalidRequest(
                "max_tokens must be greater than zero",
            ));
        }
        let capacity =
            prompt_tokens
                .len()
                .checked_add(params.max_tokens)
                .ok_or(Error::InvalidRequest(
                    "prompt plus output length overflows usize",
                ))?;
        let blocks_needed = capacity.div_ceil(self.config.block_size);
        if blocks_needed > self.config.num_kv_blocks {
            return Err(Error::CapacityExceeded {
                required: blocks_needed,
                available: self.config.num_kv_blocks,
            });
        }
        let id = self.next_id;
        self.next_id = id.checked_add(1).ok_or(Error::RequestIdExhausted)?;
        self.scheduler.waiting.push_back(Sequence {
            id,
            tokens: prompt_tokens,
            max_tokens: params.max_tokens,
            generated: 0,
            blocks_needed,
            blocks: Vec::new(),
        });
        Ok(id)
    }

    /// Admit waiting work, then generate one token per active request.
    /// Any backend error aborts all requests, releases reservations and permanently
    /// fails this engine. Tokens produced earlier in that failing step are discarded.
    pub fn step(&mut self) -> Result<Vec<GenerationEvent>, Error> {
        if self.failed {
            return Err(Error::EngineFailed);
        }
        self.scheduler
            .admit(&mut self.blocks, self.config.max_running_requests);
        let mut events = Vec::with_capacity(self.scheduler.running.len());
        for index in 0..self.scheduler.running.len() {
            let request = &self.scheduler.running[index];
            let token = match self.backend.next_token(request.id, &request.tokens) {
                Ok(token) => token,
                Err(error) => {
                    self.scheduler.abort(&mut self.blocks);
                    self.failed = true;
                    return Err(error);
                }
            };
            let request = &mut self.scheduler.running[index];
            request.tokens.push(token);
            request.generated += 1;
            events.push(GenerationEvent {
                request_id: request.id,
                token_id: token,
                finished: request.generated == request.max_tokens,
            });
        }
        self.scheduler.reap(&mut self.blocks);
        Ok(events)
    }

    pub fn is_finished(&self) -> bool {
        self.scheduler.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MockBackend;

    fn params(max_tokens: usize) -> SamplingParams {
        SamplingParams { max_tokens }
    }

    #[test]
    fn insufficient_free_capacity_queues_then_recycles() {
        let config = Config {
            block_size: 2,
            num_kv_blocks: 2,
            max_running_requests: 2,
        };
        let mut engine = Engine::new(config, MockBackend).unwrap();
        let first = engine.add_request(vec![10], params(2)).unwrap();
        let second = engine.add_request(vec![20], params(2)).unwrap();
        for (id, token, finished) in [
            (first, 11, false),
            (first, 12, true),
            (second, 21, false),
            (second, 22, true),
        ] {
            assert_eq!(
                engine.step().unwrap(),
                vec![GenerationEvent {
                    request_id: id,
                    token_id: token,
                    finished
                }]
            );
        }
        assert!(engine.is_finished());
        engine.add_request(vec![30], params(2)).unwrap();
        assert_eq!(engine.step().unwrap().len(), 1);
        assert!(engine.step().unwrap()[0].finished);
        assert!(engine.step().unwrap().is_empty());
    }

    #[test]
    fn active_requests_each_emit_one_token() {
        let mut engine = Engine::new(Config::default(), MockBackend).unwrap();
        for token in 1..=3 {
            engine.add_request(vec![token], params(2)).unwrap();
        }
        let first = engine.step().unwrap();
        assert_eq!(first.len(), 3);
        assert!(first.iter().all(|event| !event.finished));
        let second = engine.step().unwrap();
        assert_eq!(second.len(), 3);
        assert!(second.iter().all(|event| event.finished));
        assert!(engine.is_finished());
    }

    #[test]
    fn invalid_input_and_overflow_are_rejected() {
        for config in [
            Config {
                block_size: 0,
                ..Config::default()
            },
            Config {
                num_kv_blocks: 0,
                ..Config::default()
            },
            Config {
                max_running_requests: 0,
                ..Config::default()
            },
            Config {
                block_size: usize::MAX,
                num_kv_blocks: 2,
                max_running_requests: 1,
            },
        ] {
            assert!(config.validate().is_err());
        }
        let mut engine = Engine::new(
            Config {
                block_size: 2,
                num_kv_blocks: 1,
                max_running_requests: 1,
            },
            MockBackend,
        )
        .unwrap();
        assert!(matches!(
            engine.add_request(vec![], params(1)),
            Err(Error::InvalidRequest(_))
        ));
        assert!(matches!(
            engine.add_request(vec![1], params(0)),
            Err(Error::InvalidRequest(_))
        ));
        assert!(matches!(
            engine.add_request(vec![1], params(usize::MAX)),
            Err(Error::InvalidRequest(_))
        ));
        assert_eq!(
            engine.add_request(vec![1], params(2)),
            Err(Error::CapacityExceeded {
                required: 2,
                available: 1
            })
        );
        engine.next_id = u64::MAX;
        assert_eq!(
            engine.add_request(vec![1], params(1)),
            Err(Error::RequestIdExhausted)
        );
        assert!(engine.is_finished());
    }

    #[test]
    fn backend_failure_aborts_and_releases_every_reservation() {
        struct Broken;
        impl Backend for Broken {
            fn next_token(&mut self, _: RequestId, _: &[u32]) -> Result<u32, Error> {
                Err(Error::Backend("test failure".into()))
            }
        }
        let mut engine = Engine::new(
            Config {
                block_size: 2,
                num_kv_blocks: 2,
                max_running_requests: 2,
            },
            Broken,
        )
        .unwrap();
        for _ in 0..3 {
            engine.add_request(vec![1], params(1)).unwrap();
        }
        assert!(matches!(engine.step(), Err(Error::Backend(_))));
        assert!(engine.is_finished());
        assert_eq!(engine.blocks.reserve(2).unwrap().len(), 2);
        assert_eq!(engine.step(), Err(Error::EngineFailed));
        assert_eq!(
            engine.add_request(vec![1], params(1)),
            Err(Error::EngineFailed)
        );
    }
}
