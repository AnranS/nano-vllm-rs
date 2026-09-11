/// Logical ID bookkeeping only: there is no GPU memory or paged attention.
pub(super) struct BlockManager {
    free: Vec<usize>,
}

impl BlockManager {
    pub fn new(capacity: usize) -> Self {
        Self {
            free: (0..capacity).rev().collect(),
        }
    }

    pub fn reserve(&mut self, count: usize) -> Option<Vec<usize>> {
        if count > self.free.len() {
            return None;
        }
        Some(self.free.split_off(self.free.len() - count))
    }

    pub fn release(&mut self, blocks: &mut Vec<usize>) {
        self.free.append(blocks);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservations_are_unique_and_recycled() {
        let mut manager = BlockManager::new(3);
        let mut first = manager.reserve(2).unwrap();
        let second = manager.reserve(1).unwrap();
        let unique: std::collections::HashSet<_> = first.iter().chain(&second).collect();
        assert_eq!(unique.len(), 3);
        assert!(manager.reserve(1).is_none());
        let old = first.clone();
        manager.release(&mut first);
        assert!(first.is_empty());
        assert_eq!(manager.reserve(2).unwrap(), old);
    }
}
