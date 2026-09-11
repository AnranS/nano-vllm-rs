//! Continuous batching and reference-counted paged KV-cache scheduling.
//!
//! Physical KV tensors belong to the model runner. This module controls their
//! block tables, sharing only fully computed blocks with exactly matching token
//! prefixes. Allocation pressure preempts a sequence and later recomputes its
//! retained tokens; it never silently truncates a request.

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::time::Instant;

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

/// Flattened input for one model forward pass. Sequence IDs index block-table
/// rows, not user request IDs. Sampling arrays correspond to last_token_indices.
#[derive(Debug, Clone, Default)]
pub struct Batch {
    pub token_ids: Vec<i32>,
    pub positions: Vec<i32>,
    pub sequence_ids: Vec<i32>,
    pub slot_mapping: Vec<i32>,
    pub block_tables: Vec<i32>,
    pub table_stride: usize,
    pub last_token_indices: Vec<i32>,
    pub temperatures: Vec<f32>,
    pub seeds: Vec<u64>,
    pub is_decode: bool,
}

impl Batch {
    pub fn sequence_count(&self) -> usize {
        self.block_tables
            .len()
            .checked_div(self.table_stride)
            .unwrap_or(0)
    }
}

pub trait ModelRunner {
    /// Return one sampled token per last_token_indices entry. Intermediate
    /// prefill chunks can have no sampling entries while still writing KV.
    fn run(&mut self, batch: &Batch) -> Result<Vec<u32>>;

    /// Wait for device work so timings and cache publication are accurate.
    fn synchronize(&mut self) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Request {
    pub id: u64,
    pub prompt_token_ids: Vec<u32>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub ignore_eos: bool,
}

impl Default for Request {
    fn default() -> Self {
        Self {
            id: 0,
            prompt_token_ids: Vec::new(),
            max_tokens: 32,
            temperature: 0.0,
            ignore_eos: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SchedulerConfig {
    pub block_size: usize,
    pub num_blocks: usize,
    pub max_sequences: usize,
    pub max_model_len: usize,
    pub max_batch_tokens: usize,
    pub prefix_caching: bool,
    pub eos_token_id: Option<u32>,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            block_size: 256,
            num_blocks: 128,
            max_sequences: 32,
            max_model_len: 4096,
            max_batch_tokens: 1024,
            prefix_caching: true,
            eos_token_id: None,
        }
    }
}

impl SchedulerConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.block_size > 0, "block_size must be positive");
        ensure!(self.num_blocks > 0, "num_blocks must be positive");
        ensure!(self.max_sequences > 0, "max_sequences must be positive");
        ensure!(self.max_model_len > 0, "max_model_len must be positive");
        ensure!(
            self.max_batch_tokens > 0,
            "max_batch_tokens must be positive"
        );
        let slots = self.block_size.checked_mul(self.num_blocks);
        ensure!(
            slots.is_some_and(|n| n <= i32::MAX as usize),
            "KV slot indices must fit in i32"
        );
        ensure!(
            self.max_model_len <= i32::MAX as usize
                && self.max_sequences <= i32::MAX as usize
                && self.max_batch_tokens <= i32::MAX as usize,
            "batch dimensions must fit in i32"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Length,
    Eos,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestOutput {
    pub id: u64,
    /// Newly generated tokens, including a terminating EOS token if present.
    pub token_ids: Vec<u32>,
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    /// Starts when the complete generate() request batch is submitted; includes queueing.
    pub ttft_ms: f64,
    pub latency_ms: f64,
    pub inter_token_latencies_ms: Vec<f64>,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunStats {
    pub elapsed_seconds: f64,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub output_tokens_per_second: f64,
    pub total_requests: usize,
    /// Reused KV tokens, including hits when a preempted request is resumed.
    pub cached_tokens: usize,
    pub cache_hits: usize,
    /// Executed prefill tokens; includes recomputation after preemption.
    pub prefill_tokens: usize,
    pub decode_tokens: usize,
    pub prefill_steps: usize,
    pub decode_steps: usize,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
    pub preemptions: usize,
    pub peak_kv_blocks: usize,
    pub mean_ttft_ms: f64,
    pub mean_latency_ms: f64,
    pub mean_inter_token_latency_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunOutput {
    /// Preserves input request order, irrespective of scheduling/completion order.
    pub outputs: Vec<RequestOutput>,
    pub stats: RunStats,
}

#[derive(Debug)]
struct CachedPrefix {
    hash: u64,
    // Exact prefix comparison prevents hash collisions and identical blocks
    // following different contexts from sharing incorrect KV. Keys are removed
    // on eviction; memory is bounded by num_blocks * max_model_len tokens.
    tokens: Vec<u32>,
}

#[derive(Debug, Default)]
struct PhysicalBlock {
    references: usize,
    prefix: Option<CachedPrefix>,
    last_used: u64,
}

#[derive(Debug)]
struct BlockPool {
    blocks: Vec<PhysicalBlock>,
    by_hash: HashMap<u64, Vec<usize>>,
    clock: u64,
}

impl BlockPool {
    fn new(count: usize) -> Self {
        Self {
            blocks: (0..count).map(|_| PhysicalBlock::default()).collect(),
            by_hash: HashMap::new(),
            clock: 0,
        }
    }

    fn prefix_hash(tokens: &[u32]) -> u64 {
        let mut hasher = DefaultHasher::new();
        tokens.hash(&mut hasher);
        hasher.finish()
    }

    fn touch(&mut self, block: usize) {
        self.clock = self.clock.saturating_add(1);
        self.blocks[block].last_used = self.clock;
    }

    fn available(&self) -> usize {
        self.blocks.iter().filter(|b| b.references == 0).count()
    }

    fn in_use(&self) -> usize {
        self.blocks.len() - self.available()
    }

    fn invalidate(&mut self, block: usize) {
        if let Some(prefix) = self.blocks[block].prefix.take()
            && let Some(entries) = self.by_hash.get_mut(&prefix.hash)
        {
            entries.retain(|&id| id != block);
            if entries.is_empty() {
                self.by_hash.remove(&prefix.hash);
            }
        }
    }

    fn allocate(&mut self) -> Option<usize> {
        let block = self
            .blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| b.references == 0)
            // Empty blocks first; otherwise evict the least recently used key.
            .min_by_key(|(_, b)| (b.prefix.is_some(), b.last_used))
            .map(|(id, _)| id)?;
        self.invalidate(block);
        self.blocks[block].references = 1;
        self.touch(block);
        Some(block)
    }

    fn release(&mut self, block: usize) {
        debug_assert!(self.blocks[block].references > 0);
        self.blocks[block].references -= 1;
        self.touch(block);
    }

    fn find(&self, tokens: &[u32]) -> Option<usize> {
        let hash = Self::prefix_hash(tokens);
        self.by_hash.get(&hash)?.iter().copied().find(|&id| {
            self.blocks[id]
                .prefix
                .as_ref()
                .is_some_and(|key| key.tokens == tokens)
        })
    }

    fn find_and_retain(&mut self, tokens: &[u32]) -> Option<usize> {
        let block = self.find(tokens)?;
        self.blocks[block].references += 1;
        self.touch(block);
        Some(block)
    }

    fn publish(&mut self, block: usize, prefix: &[u32]) {
        if self.blocks[block].prefix.is_some() {
            debug_assert_eq!(self.blocks[block].prefix.as_ref().unwrap().tokens, prefix);
            return;
        }
        let hash = Self::prefix_hash(prefix);
        self.blocks[block].prefix = Some(CachedPrefix {
            hash,
            tokens: prefix.to_vec(),
        });
        self.by_hash.entry(hash).or_default().push(block);
        self.touch(block);
    }

    fn clear_cache(&mut self) {
        self.by_hash.clear();
        for block in &mut self.blocks {
            block.prefix = None;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Waiting,
    Active,
    Finished,
}

struct Sequence<'a> {
    request: &'a Request,
    tokens: Vec<u32>,
    computed: usize,
    blocks: Vec<usize>,
    status: Status,
    cached_tokens: usize,
    first_token_ms: Option<f64>,
    previous_token_ms: Option<f64>,
    latency_ms: f64,
    gaps_ms: Vec<f64>,
    finish_reason: FinishReason,
}

impl<'a> Sequence<'a> {
    fn new(request: &'a Request) -> Self {
        Self {
            request,
            tokens: request.prompt_token_ids.clone(),
            computed: 0,
            blocks: Vec::new(),
            status: Status::Waiting,
            cached_tokens: 0,
            first_token_ms: None,
            previous_token_ms: None,
            latency_ms: 0.0,
            gaps_ms: Vec::new(),
            finish_reason: FinishReason::Length,
        }
    }

    fn generated(&self) -> usize {
        self.tokens.len() - self.request.prompt_token_ids.len()
    }

    fn is_decode(&self) -> bool {
        self.generated() > 0 && self.computed + 1 == self.tokens.len()
    }

    fn release(&mut self, pool: &mut BlockPool) {
        for block in self.blocks.drain(..) {
            pool.release(block);
        }
        self.computed = 0;
    }
}

struct Segment {
    sequence: usize,
    start: usize,
    end: usize,
    sample: bool,
}

pub struct LlmEngine<R: ModelRunner> {
    config: SchedulerConfig,
    runner: R,
    blocks: BlockPool,
    failed: bool,
}

impl<R: ModelRunner> LlmEngine<R> {
    pub fn new(config: SchedulerConfig, runner: R) -> Result<Self> {
        config.validate()?;
        let blocks = BlockPool::new(config.num_blocks);
        Ok(Self {
            config,
            runner,
            blocks,
            failed: false,
        })
    }

    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    pub fn runner_mut(&mut self) -> &mut R {
        &mut self.runner
    }

    /// Invalidate reusable prefixes between cold-cache benchmark trials.
    pub fn clear_prefix_cache(&mut self) -> Result<()> {
        ensure!(self.blocks.in_use() == 0, "requests still hold KV blocks");
        self.blocks.clear_cache();
        Ok(())
    }

    pub fn generate(&mut self, requests: &[Request], seed: u64) -> Result<RunOutput> {
        ensure!(
            !self.failed,
            "engine is unusable after a model runner failure"
        );
        self.validate_requests(requests)?;
        self.runner.synchronize()?;
        let started = Instant::now();
        let mut sequences: Vec<_> = requests.iter().map(Sequence::new).collect();
        let result = self.execute(&mut sequences, seed, started);
        // Cleanup is unconditional, including partially executed model failures.
        for sequence in &mut sequences {
            sequence.release(&mut self.blocks);
        }
        if result.is_err() {
            self.failed = true;
            self.blocks.clear_cache();
        }
        result
    }

    fn validate_requests(&self, requests: &[Request]) -> Result<()> {
        let mut ids = HashSet::new();
        for request in requests {
            ensure!(
                ids.insert(request.id),
                "duplicate request ID {}",
                request.id
            );
            ensure!(
                !request.prompt_token_ids.is_empty(),
                "prompt must not be empty"
            );
            ensure!(request.max_tokens > 0, "max_tokens must be positive");
            ensure!(
                request.temperature.is_finite() && request.temperature >= 0.0,
                "temperature must be finite and nonnegative"
            );
            ensure!(
                request
                    .prompt_token_ids
                    .iter()
                    .all(|&t| t <= i32::MAX as u32),
                "token IDs must fit in i32"
            );
            let length = request
                .prompt_token_ids
                .len()
                .checked_add(request.max_tokens)
                .ok_or_else(|| anyhow::anyhow!("request token length overflow"))?;
            ensure!(
                length <= self.config.max_model_len,
                "request {} needs {length} context tokens, exceeding max_model_len {}",
                request.id,
                self.config.max_model_len
            );
            // The final sampled token is returned without a further KV write.
            let needed = (length - 1).div_ceil(self.config.block_size);
            ensure!(
                needed <= self.config.num_blocks,
                "request {} needs up to {needed} KV blocks but only {} exist",
                request.id,
                self.config.num_blocks
            );
        }
        Ok(())
    }

    /// Blocks needed to compute the current retained tokens. Already pinned
    /// matching prefixes cost no new physical capacity; unreferenced cached
    /// blocks still have to be removed from the available pool.
    fn admission_blocks(&self, sequence: &Sequence<'_>) -> usize {
        let mut needed = sequence.tokens.len().div_ceil(self.config.block_size);
        if self.config.prefix_caching {
            let reusable = (sequence.tokens.len() - 1) / self.config.block_size;
            for logical in 0..reusable {
                let end = (logical + 1) * self.config.block_size;
                let Some(block) = self.blocks.find(&sequence.tokens[..end]) else {
                    break;
                };
                if self.blocks.blocks[block].references > 0 {
                    needed -= 1;
                }
            }
        }
        needed
    }

    fn admit(&mut self, sequence: &mut Sequence<'_>, stats: &mut RunStats) -> Result<()> {
        debug_assert!(sequence.blocks.is_empty());
        sequence.status = Status::Active;
        // Always run at least the final token to recover logits, including when
        // the entire prompt is an exact cached block multiple.
        let reusable = if self.config.prefix_caching {
            (sequence.tokens.len() - 1) / self.config.block_size
        } else {
            0
        };
        for logical in 0..reusable {
            let end = (logical + 1) * self.config.block_size;
            let Some(block) = self.blocks.find_and_retain(&sequence.tokens[..end]) else {
                break;
            };
            sequence.blocks.push(block);
            sequence.computed = end;
            sequence.cached_tokens += self.config.block_size;
            stats.cached_tokens += self.config.block_size;
            stats.cache_hits += 1;
        }
        // Reserve only this prefill's tokens, including chunks not executed yet.
        // A newly admitted prompt must not evict an established decode request.
        // Future generated-token blocks continue to grow dynamically.
        while sequence.blocks.len() < sequence.tokens.len().div_ceil(self.config.block_size) {
            let block = self
                .blocks
                .allocate()
                .ok_or_else(|| anyhow::anyhow!("prefill admission capacity invariant violated"))?;
            sequence.blocks.push(block);
        }
        stats.peak_kv_blocks = stats.peak_kv_blocks.max(self.blocks.in_use());
        Ok(())
    }

    fn execute(
        &mut self,
        sequences: &mut [Sequence<'_>],
        seed: u64,
        started: Instant,
    ) -> Result<RunOutput> {
        let mut waiting: VecDeque<usize> = (0..sequences.len()).collect();
        let mut active: Vec<usize> = Vec::new();
        let mut finished = 0;
        let mut stats = RunStats {
            input_tokens: sequences.iter().map(|s| s.tokens.len()).sum(),
            total_requests: sequences.len(),
            ..RunStats::default()
        };

        while finished < sequences.len() {
            while active.len() < self.config.max_sequences {
                let Some(&index) = waiting.front() else {
                    break;
                };
                let needed = self.admission_blocks(&sequences[index]);
                // Protect allocations already needed by the next decode step.
                let decode_growth: usize = active
                    .iter()
                    .map(|&i| {
                        sequences[i]
                            .tokens
                            .len()
                            .div_ceil(self.config.block_size)
                            .saturating_sub(sequences[i].blocks.len())
                    })
                    .sum();
                if needed + decode_growth > self.blocks.available() {
                    break;
                }
                waiting.pop_front();
                self.admit(&mut sequences[index], &mut stats)?;
                active.push(index);
            }
            ensure!(!active.is_empty(), "scheduler made no progress");

            // Prefill and decode use separate forwards. Prefill is chunked by
            // the same hard token budget, so a long prompt cannot form an
            // unbounded forward pass.
            let is_decode = active.iter().all(|&i| sequences[i].is_decode());
            let candidates: Vec<usize> = active
                .iter()
                .copied()
                .filter(|&i| sequences[i].is_decode() == is_decode)
                .collect();
            let mut segments: Vec<Segment> = Vec::new();
            let mut budget = self.config.max_batch_tokens;

            for index in candidates {
                if budget == 0 || sequences[index].status != Status::Active {
                    continue;
                }
                let start = sequences[index].computed;
                let mut end = sequences[index].tokens.len().min(start + budget);
                let desired = end.div_ceil(self.config.block_size);
                let extra = desired.saturating_sub(sequences[index].blocks.len());

                while self.blocks.available() < extra {
                    // Already constructed batch rows must remain pinned. Favor
                    // older requests, preempting the youngest unprotected one.
                    let victim = active.iter().rev().copied().find(|&other| {
                        other != index
                            && !sequences[other].blocks.is_empty()
                            && !segments.iter().any(|s| s.sequence == other)
                    });
                    let Some(victim) = victim else {
                        break;
                    };
                    sequences[victim].release(&mut self.blocks);
                    sequences[victim].status = Status::Waiting;
                    active.retain(|&i| i != victim);
                    waiting.push_front(victim);
                    stats.preemptions += 1;
                    // Capacity-aware admission restores victims only when their
                    // complete retained prefill fits, avoiding replay thrashing.
                }

                let capacity = (sequences[index].blocks.len() + self.blocks.available())
                    * self.config.block_size;
                end = end.min(capacity);
                if end <= start {
                    continue;
                }
                while sequences[index].blocks.len() < end.div_ceil(self.config.block_size) {
                    let block = self
                        .blocks
                        .allocate()
                        .ok_or_else(|| anyhow::anyhow!("KV allocation invariant violated"))?;
                    sequences[index].blocks.push(block);
                }
                stats.peak_kv_blocks = stats.peak_kv_blocks.max(self.blocks.in_use());
                segments.push(Segment {
                    sequence: index,
                    start,
                    end,
                    sample: end == sequences[index].tokens.len(),
                });
                budget -= end - start;
            }

            ensure!(
                !segments.is_empty(),
                "KV pressure prevented scheduler progress"
            );
            let batch = self.make_batch(sequences, &segments, seed, is_decode);
            let forward_started = Instant::now();
            let sampled = self.runner.run(&batch)?;
            self.runner.synchronize()?;
            let forward_seconds = forward_started.elapsed().as_secs_f64();
            ensure!(
                sampled.len() == batch.last_token_indices.len(),
                "model returned {} samples for {} sampling rows",
                sampled.len(),
                batch.last_token_indices.len()
            );
            ensure!(
                sampled.iter().all(|&token| token <= i32::MAX as u32),
                "model returned a token ID that does not fit in i32"
            );

            if is_decode {
                stats.decode_steps += 1;
                stats.decode_tokens += batch.token_ids.len();
                stats.decode_seconds += forward_seconds;
            } else {
                stats.prefill_steps += 1;
                stats.prefill_tokens += batch.token_ids.len();
                stats.prefill_seconds += forward_seconds;
            }
            let now_ms = started.elapsed().as_secs_f64() * 1000.0;
            let mut sample_index = 0;
            for segment in segments {
                let sequence = &mut sequences[segment.sequence];
                sequence.computed = segment.end;
                if self.config.prefix_caching {
                    let first = segment.start / self.config.block_size;
                    let complete = sequence.computed / self.config.block_size;
                    for logical in first..complete {
                        let prefix_end = (logical + 1) * self.config.block_size;
                        self.blocks
                            .publish(sequence.blocks[logical], &sequence.tokens[..prefix_end]);
                    }
                }
                if !segment.sample {
                    continue;
                }
                let token = sampled[sample_index];
                sample_index += 1;
                sequence.tokens.push(token);
                sequence.first_token_ms.get_or_insert(now_ms);
                if let Some(previous) = sequence.previous_token_ms.replace(now_ms) {
                    sequence.gaps_ms.push(now_ms - previous);
                }
                let eos = !sequence.request.ignore_eos && self.config.eos_token_id == Some(token);
                if eos || sequence.generated() == sequence.request.max_tokens {
                    sequence.finish_reason = if eos {
                        FinishReason::Eos
                    } else {
                        FinishReason::Length
                    };
                    sequence.latency_ms = now_ms;
                    sequence.status = Status::Finished;
                    sequence.release(&mut self.blocks);
                    finished += 1;
                }
            }
            active.retain(|&i| sequences[i].status == Status::Active);
        }

        stats.elapsed_seconds = started.elapsed().as_secs_f64();
        let outputs: Vec<_> = sequences
            .iter()
            .map(|sequence| RequestOutput {
                id: sequence.request.id,
                token_ids: sequence.tokens[sequence.request.prompt_token_ids.len()..].to_vec(),
                prompt_tokens: sequence.request.prompt_token_ids.len(),
                cached_tokens: sequence.cached_tokens,
                ttft_ms: sequence.first_token_ms.unwrap_or_default(),
                latency_ms: sequence.latency_ms,
                inter_token_latencies_ms: sequence.gaps_ms.clone(),
                finish_reason: sequence.finish_reason,
            })
            .collect();
        stats.output_tokens = outputs.iter().map(|o| o.token_ids.len()).sum();
        if stats.elapsed_seconds > 0.0 {
            stats.output_tokens_per_second = stats.output_tokens as f64 / stats.elapsed_seconds;
        }
        if !outputs.is_empty() {
            stats.mean_ttft_ms =
                outputs.iter().map(|o| o.ttft_ms).sum::<f64>() / outputs.len() as f64;
            stats.mean_latency_ms =
                outputs.iter().map(|o| o.latency_ms).sum::<f64>() / outputs.len() as f64;
        }
        let gaps: usize = outputs
            .iter()
            .map(|o| o.inter_token_latencies_ms.len())
            .sum();
        if gaps > 0 {
            stats.mean_inter_token_latency_ms = outputs
                .iter()
                .flat_map(|o| &o.inter_token_latencies_ms)
                .sum::<f64>()
                / gaps as f64;
        }
        Ok(RunOutput { outputs, stats })
    }

    fn make_batch(
        &self,
        sequences: &[Sequence<'_>],
        segments: &[Segment],
        seed: u64,
        is_decode: bool,
    ) -> Batch {
        let table_stride = segments
            .iter()
            .map(|s| sequences[s.sequence].blocks.len())
            .max()
            .unwrap_or(0);
        let mut batch = Batch {
            table_stride,
            is_decode,
            ..Batch::default()
        };
        for (row, segment) in segments.iter().enumerate() {
            let sequence = &sequences[segment.sequence];
            for position in segment.start..segment.end {
                let physical = sequence.blocks[position / self.config.block_size];
                batch.token_ids.push(sequence.tokens[position] as i32);
                batch.positions.push(position as i32);
                batch.sequence_ids.push(row as i32);
                batch.slot_mapping.push(
                    (physical * self.config.block_size + position % self.config.block_size) as i32,
                );
            }
            for logical in 0..table_stride {
                batch.block_tables.push(
                    sequence
                        .blocks
                        .get(logical)
                        .map_or(-1, |&block| block as i32),
                );
            }
            if segment.sample {
                batch
                    .last_token_indices
                    .push(batch.token_ids.len() as i32 - 1);
                batch.temperatures.push(sequence.request.temperature);
                batch.seeds.push(sampling_seed(
                    seed,
                    sequence.request.id,
                    sequence.generated(),
                ));
            }
        }
        batch
    }
}

// Per-request/per-output-step seeds are independent of admission order, batching,
// prefix-cache hits, and recomputation. No mutable RNG is advanced on a replay.
fn sampling_seed(base: u64, request_id: u64, generated: usize) -> u64 {
    fn mix(mut value: u64) -> u64 {
        value = value.wrapping_add(0x9e3779b97f4a7c15);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }
    mix(mix(base) ^ mix(request_id) ^ mix(generated as u64))
}

#[cfg(test)]
mod tests;
