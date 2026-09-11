//! Single-GPU Qwen3 inference with persistent BF16 workspaces and paged KV storage.
//!
//! Model weights are read directly from local Hugging Face safetensors files. No
//! Python runtime, downloaded code, or libtorch is involved in model execution.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use half::{bf16, f16};
use serde::{Deserialize, Serialize};

use crate::gpu::{Buffer, Gpu, Graph};
use crate::runtime::{Batch, ModelRunner};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelConfig {
    #[serde(default)]
    pub architectures: Vec<String>,
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub hidden_act: String,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    pub quantization_config: Option<serde_json::Value>,
    #[serde(default)]
    pub eos_token_id: serde_json::Value,
}

impl ModelConfig {
    pub fn from_path(path: &Path) -> Result<Self> {
        let config_path = path.join("config.json");
        let value: Self = serde_json::from_slice(
            &fs::read(&config_path)
                .with_context(|| format!("reading {}", config_path.display()))?,
        )
        .context("parsing Qwen3 config.json")?;
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.model_type == "qwen3",
            "unsupported model_type {:?}; expected qwen3",
            self.model_type
        );
        ensure!(
            self.architectures.is_empty()
                || self
                    .architectures
                    .iter()
                    .all(|name| name == "Qwen3ForCausalLM"),
            "only the Qwen3ForCausalLM architecture is supported"
        );
        ensure!(
            self.hidden_act == "silu",
            "only Qwen3 SiLU activation is supported"
        );
        ensure!(!self.attention_bias, "attention_bias=true is not supported");
        ensure!(
            !self.use_sliding_window && self.sliding_window.is_none(),
            "sliding-window attention is not supported"
        );
        ensure!(
            self.rope_scaling.is_none(),
            "rope_scaling is not supported; use an unscaled Qwen3 checkpoint"
        );
        ensure!(
            self.quantization_config.is_none(),
            "quantized checkpoints are not supported; use BF16/F16/F32 safetensors"
        );
        ensure!(
            self.num_hidden_layers > 0
                && self.hidden_size > 0
                && self.intermediate_size > 0
                && self.vocab_size > 0,
            "model dimensions must be positive"
        );
        ensure!(
            self.num_attention_heads > 0
                && self.num_key_value_heads > 0
                && self
                    .num_attention_heads
                    .is_multiple_of(self.num_key_value_heads),
            "query heads must be a positive multiple of KV heads"
        );
        ensure!(
            self.head_dim > 0 && self.head_dim <= 256 && self.head_dim.is_multiple_of(2),
            "head_dim must be even and at most 256"
        );
        ensure!(
            self.rms_norm_eps.is_finite()
                && self.rms_norm_eps > 0.0
                && self.rope_theta.is_finite()
                && self.rope_theta > 0.0,
            "RMS epsilon and RoPE theta must be positive finite values"
        );
        ensure!(
            self.max_position_embeddings > 0 && self.vocab_size <= i32::MAX as usize,
            "invalid model context or vocabulary size"
        );
        ensure!(
            [
                self.hidden_size,
                self.intermediate_size,
                self.num_hidden_layers,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.max_position_embeddings
            ]
            .iter()
            .all(|&n| n <= i32::MAX as usize),
            "model dimensions exceed 32-bit CUDA indexing"
        );
        byte_size(&[self.num_attention_heads, self.head_dim], 2)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct ModelOptions {
    pub device: i32,
    pub block_size: usize,
    pub num_blocks: usize,
    pub max_batch_tokens: usize,
    pub max_sequences: usize,
    pub max_model_len: usize,
    pub cuda_graphs: bool,
}

impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            device: 0,
            block_size: 256,
            num_blocks: 128,
            max_batch_tokens: 1024,
            max_sequences: 32,
            max_model_len: 4096,
            cuda_graphs: true,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelMemoryInfo {
    pub weights_bytes: usize,
    pub kv_cache_bytes: usize,
    pub workspace_bytes: usize,
}

struct Layer {
    input_norm: Buffer,
    qkv: Buffer,
    q_norm: Buffer,
    k_norm: Buffer,
    attention_out: Buffer,
    post_norm: Buffer,
    gate_up: Buffer,
    down: Buffer,
    k_cache: Buffer,
    v_cache: Buffer,
}

struct Workspace {
    metadata: Buffer,
    tokens: Buffer,
    positions: Buffer,
    seq_indices: Buffer,
    slots: Buffer,
    block_tables: Buffer,
    last_indices: Buffer,
    temperatures: Buffer,
    seeds: Buffer,
    samples: Buffer,
    sample_scores: Buffer,
    sample_ids: Buffer,
    rope_cache: Buffer,
    residual: Buffer,
    hidden: Buffer,
    normed: Buffer,
    qkv: Buffer,
    query: Buffer,
    attention: Buffer,
    gate_up: Buffer,
    activated: Buffer,
    gathered: Buffer,
    logits: Buffer,
}

/// Fixed, naturally aligned ranges keep kernel pointers identical across graph
/// replays. One staging upload replaces eight tiny copies and synchronizations.
struct MetadataLayout {
    tokens: Range<usize>,
    positions: Range<usize>,
    sequences: Range<usize>,
    slots: Range<usize>,
    tables: Range<usize>,
    last_indices: Range<usize>,
    temperatures: Range<usize>,
    seeds: Range<usize>,
    bytes: usize,
}

impl MetadataLayout {
    fn new(tokens: usize, sequences: usize, table_stride: usize) -> Result<Self> {
        let mut cursor = 0usize;
        let mut segment = |count: usize, width: usize| -> Result<Range<usize>> {
            let start = cursor
                .checked_add(7)
                .context("metadata alignment overflow")?
                & !7;
            cursor = start
                .checked_add(byte_size(&[count], width)?)
                .context("metadata extent overflow")?;
            Ok(start..cursor)
        };
        Ok(Self {
            tokens: segment(tokens, 4)?,
            positions: segment(tokens, 4)?,
            sequences: segment(tokens, 4)?,
            slots: segment(tokens, 4)?,
            tables: segment(
                sequences
                    .checked_mul(table_stride)
                    .context("block table size overflow")?,
                4,
            )?,
            last_indices: segment(sequences, 4)?,
            temperatures: segment(sequences, 4)?,
            seeds: segment(sequences, 8)?,
            bytes: cursor,
        })
    }

    fn pack(&self, staging: &mut [u8], batch: &Batch, sequences: usize, table_stride: usize) {
        fn integers(staging: &mut [u8], start: usize, values: &[i32]) {
            let destination = &mut staging[start..start + values.len() * 4];
            for (bytes, value) in destination.as_chunks_mut::<4>().0.iter_mut().zip(values) {
                *bytes = value.to_ne_bytes();
            }
        }
        integers(staging, self.tokens.start, &batch.token_ids);
        integers(staging, self.positions.start, &batch.positions);
        integers(staging, self.sequences.start, &batch.sequence_ids);
        integers(staging, self.slots.start, &batch.slot_mapping);
        staging[self.tables.start..self.tables.start + sequences * table_stride * 4].fill(0xff);
        for sequence in 0..sequences {
            integers(
                staging,
                self.tables.start + sequence * table_stride * 4,
                &batch.block_tables
                    [sequence * batch.table_stride..(sequence + 1) * batch.table_stride],
            );
        }
        integers(staging, self.last_indices.start, &batch.last_token_indices);
        let temperatures = &mut staging
            [self.temperatures.start..self.temperatures.start + batch.temperatures.len() * 4];
        for (bytes, value) in temperatures
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(&batch.temperatures)
        {
            *bytes = value.to_ne_bytes();
        }
        let seeds = &mut staging[self.seeds.start..self.seeds.start + batch.seeds.len() * 8];
        for (bytes, value) in seeds.as_chunks_mut::<8>().0.iter_mut().zip(&batch.seeds) {
            *bytes = value.to_ne_bytes();
        }
    }
}

/// A CUDA runner. It is intentionally not Send: CUDA context and stream ownership
/// belong to the inference thread. Graphs are dropped before their device buffers.
pub struct QwenModel {
    graphs: HashMap<usize, Graph>,
    work: Workspace,
    layers: Vec<Layer>,
    embedding: Buffer,
    final_norm: Buffer,
    untied_lm_head: Option<Buffer>,
    gpu: Gpu,
    pub config: ModelConfig,
    pub options: ModelOptions,
    memory: ModelMemoryInfo,
    table_stride: usize,
    metadata_layout: MetadataLayout,
    host_metadata: Vec<u8>,
    last_sample_count: usize,
}

impl QwenModel {
    pub fn load(path: &Path, options: ModelOptions) -> Result<Self> {
        let config = ModelConfig::from_path(path)?;
        ensure!(
            options.block_size > 0
                && options.num_blocks > 0
                && options.max_batch_tokens > 0
                && options.max_sequences > 0
                && options.max_model_len > 0,
            "all model capacities must be positive"
        );
        ensure!(
            options.max_model_len <= config.max_position_embeddings,
            "max_model_len exceeds checkpoint max_position_embeddings"
        );
        ensure!(
            options.max_model_len <= 4096,
            "this CUDA attention implementation supports max_model_len up to 4096"
        );
        ensure!(
            options.max_sequences <= options.max_batch_tokens,
            "max_sequences must not exceed max_batch_tokens"
        );
        let cache_slots = options
            .num_blocks
            .checked_mul(options.block_size)
            .context("KV slot count overflow")?;
        ensure!(
            cache_slots <= i32::MAX as usize && options.max_batch_tokens <= i32::MAX as usize,
            "configured capacity exceeds 32-bit CUDA indexing"
        );
        let table_stride = options.max_model_len.div_ceil(options.block_size);
        let gpu = Gpu::new(options.device)?;
        let mut weights = WeightStore::open(path)?;
        let mut weights_bytes = 0;
        let mut load = |name: &str, shape: &[usize]| -> Result<Buffer> {
            let bytes = weights.read_bf16(name, shape)?;
            let buffer = gpu
                .alloc(bytes.len())
                .with_context(|| format!("allocating weight {name}"))?;
            buffer.upload_bytes(&bytes)?;
            weights_bytes += bytes.len();
            Ok(buffer)
        };
        let embedding = load(
            "model.embed_tokens.weight",
            &[config.vocab_size, config.hidden_size],
        )?;
        let final_norm = load("model.norm.weight", &[config.hidden_size])?;
        let untied_lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(load(
                "lm_head.weight",
                &[config.vocab_size, config.hidden_size],
            )?)
        };
        let q_size = config.num_attention_heads * config.head_dim;
        let kv_size = config.num_key_value_heads * config.head_dim;
        let cache_bytes = byte_size(&[cache_slots, kv_size], 2)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{index}");
            let mut upload = |specs: &[(&str, Vec<usize>)]| -> Result<Buffer> {
                let mut bytes = Vec::new();
                for (suffix, shape) in specs {
                    bytes.extend(weights.read_bf16(&format!("{prefix}.{suffix}"), shape)?);
                }
                let buffer = gpu
                    .alloc(bytes.len())
                    .with_context(|| format!("allocating layer {index} weight"))?;
                buffer.upload_bytes(&bytes)?;
                weights_bytes += bytes.len();
                Ok(buffer)
            };
            layers.push(Layer {
                input_norm: upload(&[("input_layernorm.weight", vec![config.hidden_size])])?,
                qkv: upload(&[
                    ("self_attn.q_proj.weight", vec![q_size, config.hidden_size]),
                    ("self_attn.k_proj.weight", vec![kv_size, config.hidden_size]),
                    ("self_attn.v_proj.weight", vec![kv_size, config.hidden_size]),
                ])?,
                q_norm: upload(&[("self_attn.q_norm.weight", vec![config.head_dim])])?,
                k_norm: upload(&[("self_attn.k_norm.weight", vec![config.head_dim])])?,
                attention_out: upload(&[(
                    "self_attn.o_proj.weight",
                    vec![config.hidden_size, q_size],
                )])?,
                post_norm: upload(&[(
                    "post_attention_layernorm.weight",
                    vec![config.hidden_size],
                )])?,
                gate_up: upload(&[
                    (
                        "mlp.gate_proj.weight",
                        vec![config.intermediate_size, config.hidden_size],
                    ),
                    (
                        "mlp.up_proj.weight",
                        vec![config.intermediate_size, config.hidden_size],
                    ),
                ])?,
                down: upload(&[(
                    "mlp.down_proj.weight",
                    vec![config.hidden_size, config.intermediate_size],
                )])?,
                k_cache: gpu
                    .alloc(cache_bytes)
                    .context("allocating paged K cache; reduce num_blocks")?,
                v_cache: gpu
                    .alloc(cache_bytes)
                    .context("allocating paged V cache; reduce num_blocks")?,
            });
        }
        let mut workspace_bytes = 0;
        let mut alloc = |dimensions: &[usize], width: usize| -> Result<Buffer> {
            let bytes = byte_size(dimensions, width)?;
            let buffer = gpu
                .alloc(bytes)
                .context("allocating model workspace; reduce max_batch_tokens or max_sequences")?;
            workspace_bytes += bytes;
            Ok(buffer)
        };
        let t = options.max_batch_tokens;
        let s = options.max_sequences;
        let h = config.hidden_size;
        let metadata_layout = MetadataLayout::new(t, s, table_stride)?;
        let metadata = alloc(&[metadata_layout.bytes], 1)?;
        let view = |range: &Range<usize>| metadata.slice(range.start, range.len());
        let work = Workspace {
            tokens: view(&metadata_layout.tokens)?,
            positions: view(&metadata_layout.positions)?,
            seq_indices: view(&metadata_layout.sequences)?,
            slots: view(&metadata_layout.slots)?,
            block_tables: view(&metadata_layout.tables)?,
            last_indices: view(&metadata_layout.last_indices)?,
            temperatures: view(&metadata_layout.temperatures)?,
            seeds: view(&metadata_layout.seeds)?,
            metadata,
            samples: alloc(&[s], 4)?,
            sample_scores: alloc(&[s, config.vocab_size.div_ceil(2048)], 4)?,
            sample_ids: alloc(&[s, config.vocab_size.div_ceil(2048)], 4)?,
            rope_cache: alloc(&[options.max_model_len, config.head_dim], 4)?,
            residual: alloc(&[t, h], 2)?,
            hidden: alloc(&[t, h], 2)?,
            normed: alloc(&[t, h], 2)?,
            qkv: alloc(&[t, q_size + 2 * kv_size], 2)?,
            query: alloc(&[t, q_size], 2)?,
            attention: alloc(&[t, q_size], 2)?,
            gate_up: alloc(&[t, 2 * config.intermediate_size], 2)?,
            activated: alloc(&[t, config.intermediate_size], 2)?,
            gathered: alloc(&[s, h], 2)?,
            logits: alloc(&[s, config.vocab_size], 2)?,
        };
        gpu.rope_cache(
            &work.rope_cache,
            options.max_model_len,
            config.head_dim,
            config.rope_theta,
        )?;
        gpu.sync()?;
        let kv_cache_bytes = byte_size(&[cache_bytes, 2, config.num_hidden_layers], 1)?;
        Ok(Self {
            graphs: HashMap::new(),
            work,
            layers,
            embedding,
            final_norm,
            untied_lm_head,
            gpu,
            config,
            options,
            memory: ModelMemoryInfo {
                weights_bytes,
                kv_cache_bytes,
                workspace_bytes,
            },
            table_stride,
            host_metadata: vec![0; metadata_layout.bytes],
            metadata_layout,
            last_sample_count: 0,
        })
    }

    pub fn memory_usage(&self) -> ModelMemoryInfo {
        self.memory.clone()
    }

    pub fn cuda_graph_count(&self) -> usize {
        self.graphs.len()
    }

    /// Return BF16 logits converted to f32 for each sampled row of the latest run.
    /// This is a diagnostic host transfer, deliberately excluded from generation.
    pub fn last_logits(&self) -> Result<Vec<Vec<f32>>> {
        let bytes = self
            .work
            .logits
            .download_bytes(self.last_sample_count * self.config.vocab_size * 2)?;
        Ok(bytes
            .chunks_exact(self.config.vocab_size * 2)
            .map(|row| {
                row.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|pair| bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]])).to_f32())
                    .collect()
            })
            .collect())
    }

    fn forward(&self, rows: usize, sequences: usize, samples: usize) -> Result<()> {
        let c = &self.config;
        let o = self.options;
        let w = &self.work;
        let g = &self.gpu;
        let q_size = c.num_attention_heads * c.head_dim;
        let qkv_size = q_size + 2 * c.num_key_value_heads * c.head_dim;
        g.embedding(
            &w.tokens,
            &self.embedding,
            &w.residual,
            rows,
            c.hidden_size,
            c.vocab_size,
        )?;
        for (index, layer) in self.layers.iter().enumerate() {
            if index == 0 {
                g.rms_norm(
                    &w.residual,
                    &layer.input_norm,
                    &w.normed,
                    rows,
                    c.hidden_size,
                    c.rms_norm_eps,
                )?;
            } else {
                g.add_rms_norm(
                    &w.hidden,
                    &w.residual,
                    &layer.input_norm,
                    &w.normed,
                    rows,
                    c.hidden_size,
                    c.rms_norm_eps,
                )?;
            }
            g.gemm(&w.normed, &layer.qkv, &w.qkv, rows, qkv_size, c.hidden_size)?;
            g.qkv_rope_cached(
                &w.qkv,
                &layer.q_norm,
                &layer.k_norm,
                &w.positions,
                &w.slots,
                &w.query,
                &layer.k_cache,
                &layer.v_cache,
                rows,
                c.num_attention_heads,
                c.num_key_value_heads,
                c.head_dim,
                o.block_size,
                o.num_blocks,
                c.rms_norm_eps,
                &w.rope_cache,
                o.max_model_len,
            )?;
            g.paged_attention(
                &w.query,
                &layer.k_cache,
                &layer.v_cache,
                &w.positions,
                &w.seq_indices,
                &w.block_tables,
                &w.attention,
                rows,
                c.num_attention_heads,
                c.num_key_value_heads,
                c.head_dim,
                o.block_size,
                o.num_blocks,
                self.table_stride,
                sequences,
                o.max_model_len,
            )?;
            g.gemm(
                &w.attention,
                &layer.attention_out,
                &w.hidden,
                rows,
                c.hidden_size,
                q_size,
            )?;
            g.add_rms_norm(
                &w.hidden,
                &w.residual,
                &layer.post_norm,
                &w.normed,
                rows,
                c.hidden_size,
                c.rms_norm_eps,
            )?;
            g.gemm(
                &w.normed,
                &layer.gate_up,
                &w.gate_up,
                rows,
                2 * c.intermediate_size,
                c.hidden_size,
            )?;
            g.silu_mul(&w.gate_up, &w.activated, rows, c.intermediate_size)?;
            g.gemm(
                &w.activated,
                &layer.down,
                &w.hidden,
                rows,
                c.hidden_size,
                c.intermediate_size,
            )?;
        }
        if samples > 0 {
            g.add_rms_norm(
                &w.hidden,
                &w.residual,
                &self.final_norm,
                &w.normed,
                rows,
                c.hidden_size,
                c.rms_norm_eps,
            )?;
            g.gather(
                &w.normed,
                &w.last_indices,
                &w.gathered,
                samples,
                c.hidden_size,
                rows,
            )?;
            g.gemm(
                &w.gathered,
                self.untied_lm_head.as_ref().unwrap_or(&self.embedding),
                &w.logits,
                samples,
                c.vocab_size,
                c.hidden_size,
            )?;
            g.sample_parallel(
                &w.logits,
                &w.temperatures,
                &w.seeds,
                &w.samples,
                &w.sample_scores,
                &w.sample_ids,
                samples,
                c.vocab_size,
            )?;
        }
        Ok(())
    }

    fn validate_batch(&self, batch: &Batch) -> Result<(usize, usize, usize)> {
        let rows = batch.token_ids.len();
        let samples = batch.last_token_indices.len();
        ensure!(
            rows > 0 && rows <= self.options.max_batch_tokens,
            "batch token count is outside model capacity"
        );
        ensure!(
            batch.positions.len() == rows
                && batch.sequence_ids.len() == rows
                && batch.slot_mapping.len() == rows,
            "token metadata lengths differ"
        );
        ensure!(
            batch.table_stride > 0
                && batch.table_stride <= self.table_stride
                && batch.block_tables.len().is_multiple_of(batch.table_stride),
            "invalid block table shape"
        );
        let sequences = batch.block_tables.len() / batch.table_stride;
        ensure!(
            sequences > 0 && sequences <= self.options.max_sequences,
            "batch sequence count exceeds capacity"
        );
        ensure!(
            samples <= sequences
                && samples == batch.temperatures.len()
                && samples == batch.seeds.len(),
            "invalid sampling metadata lengths"
        );
        ensure!(
            batch
                .token_ids
                .iter()
                .all(|&token| token >= 0 && (token as usize) < self.config.vocab_size),
            "token id outside model vocabulary"
        );
        ensure!(
            batch
                .last_token_indices
                .iter()
                .all(|&index| index >= 0 && (index as usize) < rows),
            "sample row outside batch"
        );
        ensure!(
            batch
                .temperatures
                .iter()
                .all(|&t| t.is_finite() && t >= 0.0),
            "temperature must be finite and nonnegative"
        );
        let mut seen_slots = BTreeSet::new();
        let mut max_positions = vec![None::<usize>; sequences];
        for row in 0..rows {
            let position = batch.positions[row];
            let sequence = batch.sequence_ids[row];
            ensure!(
                position >= 0 && (position as usize) < self.options.max_model_len,
                "token position exceeds max_model_len"
            );
            ensure!(
                sequence >= 0 && (sequence as usize) < sequences,
                "token sequence index outside block tables"
            );
            let position = position as usize;
            let sequence = sequence as usize;
            let block_column = position / self.options.block_size;
            ensure!(
                block_column < batch.table_stride,
                "block table does not cover token position"
            );
            let block = batch.block_tables[sequence * batch.table_stride + block_column];
            ensure!(
                block >= 0 && (block as usize) < self.options.num_blocks,
                "invalid physical KV block"
            );
            let expected_slot =
                block as usize * self.options.block_size + position % self.options.block_size;
            ensure!(
                batch.slot_mapping[row] >= 0 && batch.slot_mapping[row] as usize == expected_slot,
                "KV slot mapping disagrees with block table"
            );
            ensure!(
                seen_slots.insert(expected_slot),
                "two input tokens write the same KV slot"
            );
            max_positions[sequence] =
                Some(max_positions[sequence].map_or(position, |old| old.max(position)));
        }
        for (sequence, max_position) in max_positions.iter().enumerate() {
            if let Some(max_position) = max_position {
                for column in 0..=max_position / self.options.block_size {
                    let block = batch.block_tables[sequence * batch.table_stride + column];
                    ensure!(
                        block >= 0 && (block as usize) < self.options.num_blocks,
                        "missing KV block in causal context"
                    );
                }
            }
        }
        Ok((rows, sequences, samples))
    }
}

impl ModelRunner for QwenModel {
    fn run(&mut self, batch: &Batch) -> Result<Vec<u32>> {
        let (rows, sequences, samples) = self.validate_batch(batch)?;
        self.last_sample_count = 0;
        self.metadata_layout
            .pack(&mut self.host_metadata, batch, sequences, self.table_stride);
        // nv_upload synchronizes before returning, so both eager launches and
        // graph replay see complete input and staging may be reused safely.
        self.work.metadata.upload_bytes(&self.host_metadata)?;
        if self.options.cuda_graphs && batch.is_decode && rows == sequences && samples == sequences
        {
            if !self.graphs.contains_key(&rows) {
                // cuBLAS may initialize internal state on first use. Warm up this
                // shape before capture; replay then overwrites the same KV slots.
                self.forward(rows, sequences, samples)?;
                self.gpu.sync()?;
                let graph = self
                    .gpu
                    .capture(|| self.forward(rows, sequences, samples))?;
                self.graphs.insert(rows, graph);
            }
            self.graphs.get(&rows).expect("graph inserted").replay()?;
        } else {
            self.forward(rows, sequences, samples)?;
        }
        self.last_sample_count = samples;
        if samples == 0 {
            self.gpu.sync()?;
            Ok(Vec::new())
        } else {
            self.work
                .samples
                .download_i32(samples)?
                .into_iter()
                .map(|id| {
                    ensure!(
                        id >= 0 && (id as usize) < self.config.vocab_size,
                        "GPU sampler returned an invalid token id"
                    );
                    Ok(id as u32)
                })
                .collect()
        }
    }

    fn synchronize(&mut self) -> Result<()> {
        self.gpu.sync()
    }
}

fn byte_size(dimensions: &[usize], width: usize) -> Result<usize> {
    dimensions.iter().try_fold(width, |n, d| {
        n.checked_mul(*d).context("tensor byte size overflow")
    })
}

#[derive(Deserialize)]
struct TensorMetadata {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

struct TensorLocation {
    shard: usize,
    offset: u64,
    length: usize,
    dtype: String,
    shape: Vec<usize>,
}

struct WeightStore {
    files: Vec<File>,
    tensors: BTreeMap<String, TensorLocation>,
}

impl WeightStore {
    // Safetensors headers hold offsets relative to the end of the JSON header.
    // Read tensor ranges individually: loading a large sharded checkpoint does
    // not require retaining every shard in host memory at once.
    fn open(path: &Path) -> Result<Self> {
        let index_path = path.join("model.safetensors.index.json");
        let filenames: BTreeSet<PathBuf> = if index_path.is_file() {
            #[derive(Deserialize)]
            struct Index {
                weight_map: BTreeMap<String, String>,
            }
            let index: Index = serde_json::from_slice(&fs::read(&index_path)?)
                .context("parsing safetensors shard index")?;
            ensure!(
                !index.weight_map.is_empty(),
                "empty safetensors shard index"
            );
            index
                .weight_map
                .values()
                .map(|name| {
                    let shard = Path::new(name);
                    ensure!(
                        shard
                            .components()
                            .all(|c| matches!(c, Component::Normal(_))),
                        "shard path must be relative and contain no parent components"
                    );
                    Ok(path.join(shard))
                })
                .collect::<Result<_>>()?
        } else {
            [path.join("model.safetensors")].into_iter().collect()
        };
        let mut store = Self {
            files: Vec::new(),
            tensors: BTreeMap::new(),
        };
        for filename in filenames {
            let mut file =
                File::open(&filename).with_context(|| format!("opening {}", filename.display()))?;
            let file_length = file.metadata()?.len();
            let mut prefix = [0u8; 8];
            file.read_exact(&mut prefix)
                .context("reading safetensors header length")?;
            let header_length = u64::from_le_bytes(prefix);
            ensure!(
                header_length > 0
                    && header_length <= 100_000_000
                    && header_length <= file_length.saturating_sub(8),
                "invalid safetensors header length in {}",
                filename.display()
            );
            let data_start = 8 + header_length;
            let mut header = vec![0u8; header_length as usize];
            file.read_exact(&mut header)?;
            let mut entries: BTreeMap<String, serde_json::Value> =
                serde_json::from_slice(&header).context("parsing safetensors metadata")?;
            entries.remove("__metadata__");
            for (name, value) in entries {
                let info: TensorMetadata = serde_json::from_value(value)
                    .with_context(|| format!("invalid metadata for {name}"))?;
                let [begin, end] = info.data_offsets;
                ensure!(
                    end >= begin && end <= file_length - data_start,
                    "tensor {name} has invalid data offsets"
                );
                let width = match info.dtype.as_str() {
                    "BF16" | "F16" => 2,
                    "F32" => 4,
                    other => bail!(
                        "unsupported tensor dtype {other} for {name}; expected BF16, F16 or F32"
                    ),
                };
                let length = byte_size(&info.shape, width)?;
                ensure!(
                    u64::try_from(length)? == end - begin,
                    "tensor {name} byte size does not match its shape"
                );
                let location = TensorLocation {
                    shard: store.files.len(),
                    offset: data_start + begin,
                    length,
                    dtype: info.dtype,
                    shape: info.shape,
                };
                ensure!(
                    store.tensors.insert(name.clone(), location).is_none(),
                    "duplicate tensor {name} across safetensors shards"
                );
            }
            store.files.push(file);
        }
        Ok(store)
    }

    fn read_bf16(&mut self, name: &str, shape: &[usize]) -> Result<Vec<u8>> {
        let tensor = self
            .tensors
            .get(name)
            .with_context(|| format!("missing checkpoint tensor {name}"))?;
        ensure!(
            tensor.shape == shape,
            "tensor {name} has shape {:?}; expected {shape:?}",
            tensor.shape
        );
        let mut bytes = vec![0u8; tensor.length];
        let file = &mut self.files[tensor.shard];
        file.seek(SeekFrom::Start(tensor.offset))?;
        file.read_exact(&mut bytes)
            .with_context(|| format!("reading tensor {name}"))?;
        match tensor.dtype.as_str() {
            "BF16" => Ok(bytes),
            "F16" => Ok(bytes
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|v| {
                    bf16::from_f32(f16::from_bits(u16::from_le_bytes([v[0], v[1]])).to_f32())
                        .to_bits()
                        .to_le_bytes()
                })
                .collect()),
            "F32" => Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|v| {
                    bf16::from_f32(f32::from_le_bytes([v[0], v[1], v[2], v[3]]))
                        .to_bits()
                        .to_le_bytes()
                })
                .collect()),
            _ => unreachable!("dtype checked while indexing"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[test]
    fn packed_metadata_preserves_values_alignment_and_clears_table_padding() {
        let layout = MetadataLayout::new(3, 2, 3).unwrap();
        let ranges = [
            &layout.tokens,
            &layout.positions,
            &layout.sequences,
            &layout.slots,
            &layout.tables,
            &layout.last_indices,
            &layout.temperatures,
            &layout.seeds,
        ];
        for range in ranges {
            assert_eq!(range.start % 8, 0);
            assert!(range.end <= layout.bytes);
        }
        for adjacent in ranges.windows(2) {
            assert!(adjacent[0].end <= adjacent[1].start);
        }
        let batch = Batch {
            token_ids: vec![5, 8, 13],
            positions: vec![0, 1, 0],
            sequence_ids: vec![0, 0, 1],
            slot_mapping: vec![20, 21, 40],
            block_tables: vec![2, 4],
            table_stride: 1,
            last_token_indices: vec![1, 2],
            temperatures: vec![0.0, 0.6],
            seeds: vec![0x0102_0304_0506_0708, u64::MAX],
            is_decode: false,
        };
        let mut staging = vec![0x55; layout.bytes];
        layout.pack(&mut staging, &batch, 2, 3);
        let decode_i32 = |range: Range<usize>| {
            staging[range]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| i32::from_ne_bytes(*bytes))
                .collect::<Vec<_>>()
        };
        assert_eq!(decode_i32(layout.tokens.clone()), batch.token_ids);
        assert_eq!(decode_i32(layout.positions.clone()), batch.positions);
        assert_eq!(
            decode_i32(layout.tables.clone()),
            vec![2, -1, -1, 4, -1, -1]
        );
        assert_eq!(
            &staging[layout.seeds.start..layout.seeds.start + 8],
            &batch.seeds[0].to_ne_bytes()
        );
        assert_eq!(
            &staging[layout.temperatures.start + 4..layout.temperatures.start + 8],
            &0.6f32.to_ne_bytes()
        );
        let smaller = Batch {
            token_ids: vec![21],
            positions: vec![0],
            sequence_ids: vec![0],
            slot_mapping: vec![70],
            block_tables: vec![7],
            table_stride: 1,
            ..Batch::default()
        };
        layout.pack(&mut staging, &smaller, 1, 3);
        assert_eq!(
            &staging[layout.tables.start + 4..layout.tables.start + 12],
            &[0xff; 8]
        );
    }

    struct TestDirectory(PathBuf);
    impl TestDirectory {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "nano-vllm-rs-weights-{}-{now}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
        fn write_shard(
            &self,
            filename: &str,
            name: &str,
            dtype: &str,
            data: &[u8],
            elements: usize,
        ) {
            let header = serde_json::to_vec(&serde_json::json!({
                name: {"dtype": dtype, "shape": [elements], "data_offsets": [0, data.len()]}
            }))
            .unwrap();
            let mut file = File::create(self.0.join(filename)).unwrap();
            file.write_all(&(header.len() as u64).to_le_bytes())
                .unwrap();
            file.write_all(&header).unwrap();
            file.write_all(data).unwrap();
        }
    }
    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn checkpoint_reader_preserves_bf16_and_rejects_shape_mismatch() {
        let dir = TestDirectory::new();
        let values = [bf16::from_f32(1.25), bf16::from_f32(-2.5)];
        let data: Vec<u8> = values
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
        dir.write_shard("model.safetensors", "weight", "BF16", &data, 2);
        let mut store = WeightStore::open(&dir.0).unwrap();
        assert_eq!(store.read_bf16("weight", &[2]).unwrap(), data);
        assert!(store.read_bf16("weight", &[1, 2]).is_err());
        assert!(store.read_bf16("missing", &[2]).is_err());
    }

    #[test]
    fn checkpoint_reader_loads_shards_and_converts_fp16_fp32() {
        let dir = TestDirectory::new();
        dir.write_shard(
            "first.safetensors",
            "first",
            "F16",
            &f16::from_f32(1.5).to_bits().to_le_bytes(),
            1,
        );
        dir.write_shard(
            "second.safetensors",
            "second",
            "F32",
            &(-3.25f32).to_le_bytes(),
            1,
        );
        fs::write(
            dir.0.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "weight_map": {"first": "first.safetensors", "second": "second.safetensors"}
            }))
            .unwrap(),
        )
        .unwrap();
        let mut store = WeightStore::open(&dir.0).unwrap();
        assert_eq!(
            store.read_bf16("first", &[1]).unwrap(),
            bf16::from_f32(1.5).to_bits().to_le_bytes()
        );
        assert_eq!(
            store.read_bf16("second", &[1]).unwrap(),
            bf16::from_f32(-3.25).to_bits().to_le_bytes()
        );
    }

    #[test]
    fn checkpoint_reader_rejects_truncation_and_unsupported_dtype() {
        let dir = TestDirectory::new();
        dir.write_shard("model.safetensors", "weight", "BF16", &[0, 0], 2);
        assert!(WeightStore::open(&dir.0).is_err());
        dir.write_shard("model.safetensors", "weight", "I8", &[0, 0], 2);
        assert!(WeightStore::open(&dir.0).is_err());
    }

    #[test]
    fn unsupported_qwen_configuration_is_rejected() {
        let mut config: ModelConfig = serde_json::from_value(serde_json::json!({
            "model_type":"qwen3", "hidden_size":1024,"intermediate_size":3072,
            "num_hidden_layers":28,"num_attention_heads":16,"num_key_value_heads":8,
            "head_dim":128,"vocab_size":151936,"max_position_embeddings":40960,
            "rms_norm_eps":1e-6,"rope_theta":1e6,"hidden_act":"silu"
        }))
        .unwrap();
        assert!(config.validate().is_ok());
        config.attention_bias = true;
        assert!(config.validate().is_err());
        config.attention_bias = false;
        config.rope_scaling = Some(serde_json::json!({"rope_type":"yarn"}));
        assert!(config.validate().is_err());
    }
}
