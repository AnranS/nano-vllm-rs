use super::block_manager::BlockManager;
use crate::sequence::Sequence;
use std::collections::VecDeque;

#[derive(Default)]
pub(super) struct Scheduler {
    pub waiting: VecDeque<Sequence>,
    pub running: Vec<Sequence>,
}

impl Scheduler {
    // FIFO admission deliberately permits head-of-line blocking. Full output
    // reservations ensure admitted requests can finish without competing for blocks.
    pub fn admit(&mut self, manager: &mut BlockManager, limit: usize) {
        while self.running.len() < limit {
            let Some(next) = self.waiting.front() else {
                break;
            };
            let Some(blocks) = manager.reserve(next.blocks_needed) else {
                break;
            };
            let mut next = self.waiting.pop_front().expect("queue front exists");
            next.blocks = blocks;
            self.running.push(next);
        }
    }

    pub fn reap(&mut self, manager: &mut BlockManager) {
        self.running.retain_mut(|request| {
            if request.generated == request.max_tokens {
                manager.release(&mut request.blocks);
                false
            } else {
                true
            }
        });
    }

    pub fn abort(&mut self, manager: &mut BlockManager) {
        for request in &mut self.running {
            manager.release(&mut request.blocks);
        }
        self.running.clear();
        self.waiting.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty()
    }
}
