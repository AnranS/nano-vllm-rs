use super::*;

/// A token-level KV simulator. Every sampled token depends on every token read
/// through the actual physical block table, exposing stale/reused KV mistakes.
struct KvRunner {
    block_size: usize,
    kv: Vec<Option<u32>>,
    batches: Vec<Batch>,
    fail_at: Option<usize>,
    eos: Option<u32>,
}

impl KvRunner {
    fn new(config: &SchedulerConfig) -> Self {
        Self {
            block_size: config.block_size,
            kv: vec![None; config.block_size * config.num_blocks],
            batches: Vec::new(),
            fail_at: None,
            eos: None,
        }
    }
}

fn reference_next(context: &[u32], temperature: f32, seed: u64) -> u32 {
    let value = context.iter().fold(7u64, |state, &token| {
        state.wrapping_mul(31).wrapping_add(u64::from(token))
    });
    let random = if temperature > 0.0 { seed } else { 0 };
    ((value ^ random) % 1000 + 1) as u32
}

impl ModelRunner for KvRunner {
    fn run(&mut self, batch: &Batch) -> Result<Vec<u32>> {
        if self.fail_at == Some(self.batches.len()) {
            anyhow::bail!("injected model failure");
        }
        assert_eq!(batch.token_ids.len(), batch.positions.len());
        assert_eq!(batch.token_ids.len(), batch.sequence_ids.len());
        assert_eq!(batch.token_ids.len(), batch.slot_mapping.len());
        assert_eq!(batch.last_token_indices.len(), batch.temperatures.len());
        assert_eq!(batch.last_token_indices.len(), batch.seeds.len());
        assert!(batch.sequence_count() > 0);
        let mut written = HashSet::new();
        for index in 0..batch.token_ids.len() {
            let slot = batch.slot_mapping[index] as usize;
            let row = batch.sequence_ids[index] as usize;
            let position = batch.positions[index] as usize;
            let physical =
                batch.block_tables[row * batch.table_stride + position / self.block_size];
            assert!(physical >= 0);
            assert_eq!(
                slot,
                physical as usize * self.block_size + position % self.block_size
            );
            assert!(
                written.insert(slot),
                "two batch rows overwrite the same KV slot"
            );
            self.kv[slot] = Some(batch.token_ids[index] as u32);
        }
        let mut samples = Vec::new();
        for (sample, &index) in batch.last_token_indices.iter().enumerate() {
            let index = index as usize;
            let row = batch.sequence_ids[index] as usize;
            let end = batch.positions[index] as usize;
            let context: Vec<u32> = (0..=end)
                .map(|position| {
                    let physical =
                        batch.block_tables[row * batch.table_stride + position / self.block_size];
                    assert!(physical >= 0);
                    self.kv[physical as usize * self.block_size + position % self.block_size]
                        .expect("attention read uninitialized KV")
                })
                .collect();
            samples.push(self.eos.unwrap_or_else(|| {
                reference_next(&context, batch.temperatures[sample], batch.seeds[sample])
            }));
        }
        self.batches.push(batch.clone());
        Ok(samples)
    }
}

fn config() -> SchedulerConfig {
    SchedulerConfig {
        block_size: 2,
        num_blocks: 32,
        max_sequences: 4,
        max_model_len: 32,
        max_batch_tokens: 16,
        prefix_caching: true,
        eos_token_id: None,
    }
}

fn request(id: u64, tokens: &[u32], max_tokens: usize) -> Request {
    Request {
        id,
        prompt_token_ids: tokens.to_vec(),
        max_tokens,
        ..Request::default()
    }
}

fn expected(request: &Request, seed: u64) -> Vec<u32> {
    let mut context = request.prompt_token_ids.clone();
    for step in 0..request.max_tokens {
        let token = reference_next(
            &context,
            request.temperature,
            sampling_seed(seed, request.id, step),
        );
        context.push(token);
    }
    context[request.prompt_token_ids.len()..].to_vec()
}

#[test]
fn mixed_lengths_match_sequential_reference_and_release_all_blocks() {
    let config = config();
    let mut engine = LlmEngine::new(config.clone(), KvRunner::new(&config)).unwrap();
    let requests = [
        request(11, &[1, 2, 3], 4),
        request(12, &[5], 2),
        request(13, &[8, 9, 10, 11, 12], 3),
        request(14, &[15, 16], 1),
        request(15, &[19, 20, 21, 22], 2),
    ];
    let result = engine.generate(&requests, 123).unwrap();
    for (output, request) in result.outputs.iter().zip(&requests) {
        assert_eq!(output.id, request.id);
        assert_eq!(output.token_ids, expected(request, 123));
        assert_eq!(
            output.inter_token_latencies_ms.len(),
            request.max_tokens - 1
        );
        assert!(output.latency_ms >= output.ttft_ms);
    }
    assert_eq!(result.stats.input_tokens, 15);
    assert_eq!(result.stats.output_tokens, 12);
    assert!(result.stats.decode_steps > 0);
    assert!(
        engine
            .runner
            .batches
            .iter()
            .any(|batch| batch.sequence_count() > 1)
    );
    assert_eq!(engine.blocks.in_use(), 0);
}

#[test]
fn chunked_prefill_obeys_token_budget_and_does_not_sample_early() {
    let config = SchedulerConfig {
        max_batch_tokens: 2,
        ..config()
    };
    let mut engine = LlmEngine::new(config.clone(), KvRunner::new(&config)).unwrap();
    let req = request(1, &[1, 2, 3, 4, 5, 6, 7], 3);
    let result = engine.generate(std::slice::from_ref(&req), 1).unwrap();
    assert_eq!(result.outputs[0].token_ids, expected(&req, 1));
    assert_eq!(result.stats.prefill_tokens, 7);
    assert_eq!(result.stats.prefill_steps, 4);
    assert!(engine.runner.batches.iter().all(|b| b.token_ids.len() <= 2));
    assert!(
        engine.runner.batches[..3]
            .iter()
            .all(|b| b.last_token_indices.is_empty())
    );
}

#[test]
fn warm_prefix_is_shared_and_final_prompt_token_is_recomputed() {
    let config = config();
    let mut engine = LlmEngine::new(config.clone(), KvRunner::new(&config)).unwrap();
    let req = request(1, &[1, 2, 3, 4, 5, 6], 1);
    let cold = engine.generate(std::slice::from_ref(&req), 1).unwrap();
    let warm_requests = [
        req.clone(),
        Request {
            id: 2,
            ..req.clone()
        },
    ];
    let warm = engine.generate(&warm_requests, 1).unwrap();
    assert_eq!(warm.stats.cached_tokens, 8);
    assert_eq!(warm.stats.prefill_tokens, 4);
    assert_eq!(cold.outputs[0].token_ids, warm.outputs[0].token_ids);
    assert_eq!(cold.outputs[0].token_ids, warm.outputs[1].token_ids);
    let batch = engine.runner.batches.last().unwrap();
    assert_eq!(batch.sequence_count(), 2);
    assert_eq!(
        batch.block_tables[0],
        batch.block_tables[batch.table_stride]
    );
    assert_eq!(
        batch.block_tables[1],
        batch.block_tables[batch.table_stride + 1]
    );
    assert_ne!(
        batch.block_tables[2],
        batch.block_tables[batch.table_stride + 2]
    );
    assert_eq!(engine.blocks.in_use(), 0);
    engine.clear_prefix_cache().unwrap();
    assert!(engine.blocks.by_hash.is_empty());
    let cleared = engine.generate(&[req], 1).unwrap();
    assert_eq!(cleared.stats.cached_tokens, 0);
}

#[test]
fn prefix_lookup_checks_entire_context_and_collision_tokens() {
    let config = config();
    let mut engine = LlmEngine::new(config.clone(), KvRunner::new(&config)).unwrap();
    engine
        .generate(&[request(1, &[1, 2, 7, 8, 9], 1)], 1)
        .unwrap();
    let different = request(2, &[3, 4, 7, 8, 9], 1);
    let output = engine
        .generate(std::slice::from_ref(&different), 1)
        .unwrap();
    assert_eq!(output.stats.cached_tokens, 0);
    assert_eq!(output.outputs[0].token_ids, expected(&different, 1));

    let mut pool = BlockPool::new(1);
    let block = pool.allocate().unwrap();
    pool.publish(block, &[1, 2]);
    pool.release(block);
    let hash = BlockPool::prefix_hash(&[3, 4]);
    pool.by_hash.insert(hash, vec![block]); // Simulated hash collision.
    assert_eq!(pool.find_and_retain(&[3, 4]), None);
}

#[test]
fn lru_eviction_clears_stale_keys_and_never_evicts_referenced_blocks() {
    let mut pool = BlockPool::new(2);
    let first = pool.allocate().unwrap();
    pool.publish(first, &[1, 2]);
    pool.release(first);
    let second = pool.allocate().unwrap();
    pool.publish(second, &[3, 4]);
    pool.release(second);
    assert_eq!(pool.find_and_retain(&[1, 2]), Some(first));
    assert_eq!(pool.find_and_retain(&[1, 2]), Some(first));
    let reused = pool.allocate().unwrap();
    assert_eq!(reused, second);
    assert_eq!(pool.find_and_retain(&[3, 4]), None);
    assert_eq!(pool.blocks[first].references, 2);
    pool.release(first);
    assert_eq!(pool.allocate(), None);
    pool.release(first);
    assert_eq!(pool.allocate(), Some(first));
    assert!(pool.by_hash.is_empty());
}

#[test]
fn capacity_pressure_preempts_and_recomputes_without_losing_output() {
    let config = SchedulerConfig {
        num_blocks: 4,
        max_sequences: 3,
        max_batch_tokens: 4,
        max_model_len: 8,
        prefix_caching: false,
        ..config()
    };
    let mut engine = LlmEngine::new(config.clone(), KvRunner::new(&config)).unwrap();
    let requests = [
        request(1, &[1, 2], 6),
        request(2, &[3, 4], 6),
        request(3, &[5, 6], 6),
    ];
    let output = engine.generate(&requests, 4).unwrap();
    assert!(output.stats.preemptions > 0);
    assert_eq!(output.stats.output_tokens, 18);
    assert!(output.stats.peak_kv_blocks <= config.num_blocks);
    for (request, output) in requests.iter().zip(output.outputs) {
        assert_eq!(output.token_ids, expected(request, 4));
    }
    assert_eq!(engine.blocks.in_use(), 0);
}

#[test]
fn a_waiting_prefill_does_not_evict_established_decode_work() {
    let config = SchedulerConfig {
        block_size: 2,
        num_blocks: 4,
        max_sequences: 2,
        max_batch_tokens: 16,
        max_model_len: 16,
        prefix_caching: false,
        ..config()
    };
    let requests = [
        request(1, &[1, 2, 3, 4], 4),
        request(2, &[5], 1),
        request(3, &[6, 7, 8, 9, 10, 11], 1),
    ];
    let mut engine = LlmEngine::new(config.clone(), KvRunner::new(&config)).unwrap();
    let output = engine.generate(&requests, 5).unwrap();
    // The third prompt needs three blocks. After request 2 finishes, only two
    // blocks are free; finishing request 1 first avoids any recomputation.
    assert_eq!(output.stats.preemptions, 0);
    assert_eq!(output.stats.prefill_tokens, 11);
    assert_eq!(output.stats.output_tokens, 6);
    assert!(output.outputs[0].latency_ms <= output.outputs[2].ttft_ms);
    assert!(
        engine
            .runner
            .batches
            .iter()
            .all(|batch| batch.sequence_count() <= config.max_sequences)
    );
    for (request, result) in requests.iter().zip(&output.outputs) {
        assert_eq!(result.token_ids, expected(request, 5));
    }
    assert_eq!(engine.blocks.in_use(), 0);
}

#[test]
fn admission_accounts_for_live_shared_prefixes_at_capacity() {
    let config = SchedulerConfig {
        block_size: 2,
        num_blocks: 4,
        max_sequences: 2,
        max_batch_tokens: 4,
        max_model_len: 8,
        ..config()
    };
    let mut engine = LlmEngine::new(config.clone(), KvRunner::new(&config)).unwrap();
    let requests = [
        request(1, &[1, 2, 3, 4, 5], 3),
        request(2, &[1, 2, 3, 4, 6], 1),
    ];
    let output = engine.generate(&requests, 0).unwrap();
    // Initially only the first three-block prefill fits. Its first four tokens
    // become reusable after one chunk; the second request then needs one block.
    assert_eq!(output.stats.cached_tokens, 4);
    assert_eq!(output.stats.prefill_tokens, 6);
    assert_eq!(output.stats.preemptions, 0);
    assert!(
        engine
            .runner
            .batches
            .iter()
            .any(|batch| batch.sequence_count() == 2)
    );
    for (request, result) in requests.iter().zip(&output.outputs) {
        assert_eq!(result.token_ids, expected(request, 0));
    }
    assert_eq!(engine.blocks.in_use(), 0);
}

#[test]
fn seed_is_independent_of_chunking_queueing_cache_and_preemption() {
    let requests: Vec<_> = (1..=5)
        .map(|id| Request {
            temperature: 0.7,
            ..request(id, &[1, 2, 3], 5)
        })
        .collect();
    for max_sequences in [1, 2, 5] {
        for max_batch_tokens in [1, 3, 10] {
            for prefix_caching in [false, true] {
                let config = SchedulerConfig {
                    num_blocks: 4,
                    max_model_len: 8,
                    max_sequences,
                    max_batch_tokens,
                    prefix_caching,
                    ..config()
                };
                let mut engine = LlmEngine::new(config.clone(), KvRunner::new(&config)).unwrap();
                for _ in 0..2 {
                    let output = engine.generate(&requests, 99).unwrap();
                    for (request, output) in requests.iter().zip(output.outputs) {
                        assert_eq!(output.token_ids, expected(request, 99));
                    }
                }
            }
        }
    }
}

#[test]
fn eos_stops_generation_and_ignore_eos_preserves_length() {
    let config = SchedulerConfig {
        eos_token_id: Some(42),
        ..config()
    };
    let mut runner = KvRunner::new(&config);
    runner.eos = Some(42);
    let mut engine = LlmEngine::new(config, runner).unwrap();
    let output = engine
        .generate(
            &[
                request(1, &[10], 4),
                Request {
                    ignore_eos: true,
                    ..request(2, &[20], 4)
                },
            ],
            0,
        )
        .unwrap();
    assert_eq!(output.outputs[0].token_ids, vec![42]);
    assert_eq!(output.outputs[0].finish_reason, FinishReason::Eos);
    assert_eq!(output.outputs[1].token_ids, vec![42; 4]);
    assert_eq!(output.outputs[1].finish_reason, FinishReason::Length);
    assert_eq!(engine.blocks.in_use(), 0);
}

#[test]
fn model_failure_releases_pins_invalidates_cache_and_poison_engine() {
    let config = config();
    let mut runner = KvRunner::new(&config);
    runner.fail_at = Some(1);
    let mut engine = LlmEngine::new(config, runner).unwrap();
    assert!(engine.generate(&[request(1, &[1, 2, 3], 4)], 0).is_err());
    assert_eq!(engine.blocks.in_use(), 0);
    assert!(engine.blocks.by_hash.is_empty());
    assert!(engine.generate(&[request(2, &[1], 1)], 0).is_err());
}

#[test]
fn invalid_requests_fail_before_running_and_empty_batch_is_valid() {
    let config = SchedulerConfig {
        num_blocks: 2,
        max_model_len: 16,
        ..config()
    };
    let mut engine = LlmEngine::new(config.clone(), KvRunner::new(&config)).unwrap();
    for request in [
        request(1, &[], 1),
        request(1, &[1], 0),
        request(1, &[1], usize::MAX),
        request(1, &[1, 2, 3, 4], 3),
        request(1, &[u32::MAX], 1),
        Request {
            temperature: f32::NAN,
            ..request(1, &[1], 1)
        },
    ] {
        assert!(engine.generate(&[request], 0).is_err());
    }
    assert!(
        engine
            .generate(&[request(1, &[1], 1), request(1, &[2], 1)], 0)
            .is_err()
    );
    assert!(engine.runner.batches.is_empty());
    let output = engine.generate(&[], 0).unwrap();
    assert!(output.outputs.is_empty());
    assert_eq!(output.stats.output_tokens, 0);
    assert!(output.stats.output_tokens_per_second.is_finite());
    for invalid in [
        SchedulerConfig {
            block_size: 0,
            ..config.clone()
        },
        SchedulerConfig {
            num_blocks: 0,
            ..config.clone()
        },
        SchedulerConfig {
            max_batch_tokens: 0,
            ..config.clone()
        },
        SchedulerConfig {
            max_sequences: 0,
            ..config.clone()
        },
        SchedulerConfig {
            num_blocks: usize::MAX,
            ..config.clone()
        },
    ] {
        assert!(invalid.validate().is_err());
    }
}
