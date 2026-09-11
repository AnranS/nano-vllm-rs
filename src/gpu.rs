//! Checked, single-threaded CUDA ownership and BF16 inference operations.
//!
//! Kernels run asynchronously on one stream. Upload/download synchronize so a
//! borrowed host slice never outlives an outstanding DMA. Buffer destruction
//! uses cudaFree; captured graphs retain every allocation they reference.
#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use anyhow::{Result, anyhow, bail, ensure};
use std::{
    cell::{Cell, RefCell},
    ffi::{CStr, c_char, c_int, c_void},
    ptr::NonNull,
    rc::Rc,
};

unsafe extern "C" {
    fn nv_stream(context: *mut c_void, output: *mut *mut c_void) -> c_int;
    fn nv_sm_count(context: *mut c_void, output: *mut c_int) -> c_int;
    fn nv_last_error() -> *const c_char;
    fn nv_create(device: c_int, output: *mut *mut c_void) -> c_int;
    fn nv_destroy(context: *mut c_void);
    fn nv_alloc(context: *mut c_void, bytes: usize, output: *mut *mut c_void) -> c_int;
    fn nv_free(context: *mut c_void, buffer: *mut c_void);
    fn nv_upload(context: *mut c_void, dst: *mut c_void, src: *const c_void, bytes: usize)
    -> c_int;
    fn nv_download(
        context: *mut c_void,
        src: *const c_void,
        dst: *mut c_void,
        bytes: usize,
    ) -> c_int;
    fn nv_sync(context: *mut c_void) -> c_int;
    fn nv_mem_info(context: *mut c_void, free: *mut usize, total: *mut usize) -> c_int;
    fn nv_capture_begin(context: *mut c_void) -> c_int;
    fn nv_capture_end(context: *mut c_void, graph: *mut *mut c_void) -> c_int;
    fn nv_capture_abort(context: *mut c_void);
    fn nv_graph_replay(context: *mut c_void, graph: *mut c_void) -> c_int;
    fn nv_graph_destroy(context: *mut c_void, graph: *mut c_void);
    fn nv_gemm(
        c: *mut c_void,
        x: *const c_void,
        w: *const c_void,
        y: *mut c_void,
        m: c_int,
        n: c_int,
        k: c_int,
    ) -> c_int;
    fn nv_embedding(
        c: *mut c_void,
        tokens: *const c_void,
        w: *const c_void,
        y: *mut c_void,
        rows: c_int,
        hidden: c_int,
        vocab: c_int,
    ) -> c_int;
    fn nv_rms(
        c: *mut c_void,
        x: *const c_void,
        w: *const c_void,
        y: *mut c_void,
        rows: c_int,
        hidden: c_int,
        eps: f32,
    ) -> c_int;
    fn nv_add_rms(
        c: *mut c_void,
        x: *const c_void,
        residual: *mut c_void,
        w: *const c_void,
        y: *mut c_void,
        rows: c_int,
        hidden: c_int,
        eps: f32,
    ) -> c_int;
    fn nv_qkv_rope(
        c: *mut c_void,
        qkv: *const c_void,
        qw: *const c_void,
        kw: *const c_void,
        positions: *const c_void,
        slots: *const c_void,
        qout: *mut c_void,
        kc: *mut c_void,
        vc: *mut c_void,
        rows: c_int,
        qh: c_int,
        kvh: c_int,
        dim: c_int,
        bs: c_int,
        blocks: c_int,
        eps: f32,
        theta: f32,
    ) -> c_int;
    fn nv_rope_cache(
        c: *mut c_void,
        cache: *mut c_void,
        max_positions: c_int,
        dim: c_int,
        theta: f32,
    ) -> c_int;
    fn nv_qkv_rope_cached(
        c: *mut c_void,
        qkv: *const c_void,
        qw: *const c_void,
        kw: *const c_void,
        positions: *const c_void,
        slots: *const c_void,
        qout: *mut c_void,
        kc: *mut c_void,
        vc: *mut c_void,
        rows: c_int,
        qh: c_int,
        kvh: c_int,
        dim: c_int,
        bs: c_int,
        blocks: c_int,
        eps: f32,
        rope_cache: *const c_void,
        max_positions: c_int,
    ) -> c_int;
    fn nv_attention(
        c: *mut c_void,
        q: *const c_void,
        kc: *const c_void,
        vc: *const c_void,
        positions: *const c_void,
        seqs: *const c_void,
        tables: *const c_void,
        out: *mut c_void,
        rows: c_int,
        qh: c_int,
        kvh: c_int,
        dim: c_int,
        bs: c_int,
        blocks: c_int,
        stride: c_int,
        batch: c_int,
        max_context: c_int,
    ) -> c_int;
    fn nv_silu(
        c: *mut c_void,
        packed: *const c_void,
        out: *mut c_void,
        rows: c_int,
        intermediate: c_int,
    ) -> c_int;
    fn nv_gather(
        c: *mut c_void,
        x: *const c_void,
        indices: *const c_void,
        out: *mut c_void,
        rows: c_int,
        hidden: c_int,
        input_rows: c_int,
    ) -> c_int;
    fn nv_sample(
        c: *mut c_void,
        logits: *const c_void,
        temps: *const c_void,
        seeds: *const c_void,
        out: *mut c_void,
        batch: c_int,
        vocab: c_int,
    ) -> c_int;
    fn nv_sample_parallel(
        c: *mut c_void,
        logits: *const c_void,
        temps: *const c_void,
        seeds: *const c_void,
        out: *mut c_void,
        partial_scores: *mut c_void,
        partial_ids: *mut c_void,
        batch: c_int,
        vocab: c_int,
    ) -> c_int;
}

#[cfg(feature = "flash-attn")]
unsafe extern "C" {
    fn nvr_flash_fwd(
        stream: *mut c_void,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        out: *mut c_void,
        cu_q: *const c_void,
        kv_lens: *const c_void,
        tables: *const c_void,
        batch: c_int,
        total_q: c_int,
        max_q: c_int,
        max_k: c_int,
        q_heads: c_int,
        kv_heads: c_int,
        block_size: c_int,
        table_stride: c_int,
        splits: c_int,
        lse: *mut c_void,
        lse_accum: *mut c_void,
        out_accum: *mut c_void,
    ) -> c_int;
    fn nvr_flash_num_splits(
        batch: c_int,
        q_heads: c_int,
        kv_heads: c_int,
        max_k: c_int,
        num_sms: c_int,
    ) -> c_int;
    fn nvr_flash_last_error() -> *const c_char;
}

fn check(code: c_int) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    // SAFETY: the shim returns a thread-local NUL-terminated error string.
    let message = unsafe { CStr::from_ptr(nv_last_error()) }.to_string_lossy();
    bail!("CUDA backend: {message} (status {code})")
}
fn dimension(value: usize) -> Result<c_int> {
    ensure!(value > 0, "tensor dimensions must be positive");
    value
        .try_into()
        .map_err(|_| anyhow!("dimension {value} exceeds i32"))
}
fn extent(dims: &[usize], element_size: usize) -> Result<usize> {
    dims.iter().try_fold(element_size, |n, &dim| {
        dimension(dim)?;
        n.checked_mul(dim)
            .ok_or_else(|| anyhow!("tensor byte count overflow"))
    })
}
fn positive(value: f32, name: &str) -> Result<()> {
    ensure!(
        value.is_finite() && value > 0.0,
        "{name} must be positive and finite"
    );
    Ok(())
}

struct Context {
    handle: NonNull<c_void>,
    capturing: Cell<bool>,
    captured: RefCell<Vec<Buffer>>,
}
impl Context {
    fn ptr(&self) -> *mut c_void {
        self.handle.as_ptr()
    }
    fn outside_capture(&self) -> Result<()> {
        ensure!(
            !self.capturing.get(),
            "allocation, transfer and synchronization are forbidden during graph capture"
        );
        Ok(())
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: the Rc owners of all allocations/graphs have gone away.
        unsafe { nv_destroy(self.ptr()) }
    }
}
struct Allocation {
    context: Rc<Context>,
    ptr: NonNull<c_void>,
    bytes: usize,
}
impl Drop for Allocation {
    fn drop(&mut self) {
        // SAFETY: this is the sole owner; cudaFree waits for outstanding users.
        unsafe { nv_free(self.context.ptr(), self.ptr.as_ptr()) }
    }
}

/// Owned device bytes. Clones share an allocation; kernel calls check aliases.
#[derive(Clone)]
pub struct Buffer {
    allocation: Rc<Allocation>,
    offset: usize,
    bytes: usize,
}
impl Buffer {
    fn ptr(&self) -> *mut c_void {
        self.allocation.ptr.as_ptr().wrapping_byte_add(self.offset)
    }
    pub fn len(&self) -> usize {
        self.bytes
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn bounds(&self, bytes: usize) -> Result<()> {
        ensure!(
            bytes <= self.len(),
            "buffer needs {bytes} bytes, allocation has {}",
            self.len()
        );
        Ok(())
    }
    /// A checked device-byte view. The original allocation remains alive through
    /// this view and any graph that captures it; no host memory is borrowed.
    pub(crate) fn slice(&self, offset: usize, bytes: usize) -> Result<Self> {
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| anyhow!("buffer view extent overflow"))?;
        self.bounds(end)?;
        let offset = self
            .offset
            .checked_add(offset)
            .ok_or_else(|| anyhow!("buffer view offset overflow"))?;
        ensure!(
            offset <= self.allocation.bytes && bytes <= self.allocation.bytes - offset,
            "buffer view exceeds allocation"
        );
        Ok(Self {
            allocation: self.allocation.clone(),
            offset,
            bytes,
        })
    }
    fn overlaps(&self, other: &Self) -> bool {
        self.bytes > 0
            && other.bytes > 0
            && Rc::ptr_eq(&self.allocation, &other.allocation)
            && self.offset < other.offset + other.bytes
            && other.offset < self.offset + self.bytes
    }
    pub fn upload_bytes(&self, values: &[u8]) -> Result<()> {
        self.upload(values)
    }
    pub fn upload_i32(&self, values: &[i32]) -> Result<()> {
        self.upload(values)
    }
    pub fn upload_u64(&self, values: &[u64]) -> Result<()> {
        self.upload(values)
    }
    pub fn upload_f32(&self, values: &[f32]) -> Result<()> {
        self.upload(values)
    }
    fn upload<T: Copy>(&self, values: &[T]) -> Result<()> {
        self.allocation.context.outside_capture()?;
        let bytes = std::mem::size_of_val(values);
        self.bounds(bytes)?;
        if bytes == 0 {
            return Ok(());
        }
        // SAFETY: checked allocation bounds; nv_upload synchronizes before return.
        check(unsafe {
            nv_upload(
                self.allocation.context.ptr(),
                self.ptr(),
                values.as_ptr().cast(),
                bytes,
            )
        })
    }
    pub fn download_bytes(&self, len: usize) -> Result<Vec<u8>> {
        self.download(len)
    }
    pub fn download_i32(&self, len: usize) -> Result<Vec<i32>> {
        self.download(len)
    }
    pub fn download_u64(&self, len: usize) -> Result<Vec<u64>> {
        self.download(len)
    }
    pub fn download_f32(&self, len: usize) -> Result<Vec<f32>> {
        self.download(len)
    }
    fn download<T: Default + Copy>(&self, len: usize) -> Result<Vec<T>> {
        self.allocation.context.outside_capture()?;
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| anyhow!("download byte count overflow"))?;
        self.bounds(bytes)?;
        let mut result = vec![T::default(); len];
        if bytes != 0 {
            // SAFETY: result is initialized and lives through the synchronized copy.
            check(unsafe {
                nv_download(
                    self.allocation.context.ptr(),
                    self.ptr(),
                    result.as_mut_ptr().cast(),
                    bytes,
                )
            })?;
        }
        Ok(result)
    }
}

/// One CUDA stream and cuBLAS handle. Rc deliberately makes this !Send/!Sync.
#[derive(Clone)]
pub struct Gpu {
    context: Rc<Context>,
}
impl Gpu {
    pub fn new(device: i32) -> Result<Self> {
        ensure!(device >= 0, "CUDA device index must be nonnegative");
        let mut ptr = std::ptr::null_mut();
        // SAFETY: output points to initialized writable storage.
        check(unsafe { nv_create(device, &mut ptr) })?;
        Ok(Self {
            context: Rc::new(Context {
                handle: NonNull::new(ptr).ok_or_else(|| anyhow!("CUDA returned a null context"))?,
                capturing: Cell::new(false),
                captured: RefCell::new(Vec::new()),
            }),
        })
    }
    fn ptr(&self) -> *mut c_void {
        self.context.ptr()
    }
    pub fn alloc(&self, bytes: usize) -> Result<Buffer> {
        self.context.outside_capture()?;
        ensure!(
            bytes > 0 && bytes <= isize::MAX as usize,
            "invalid CUDA allocation size {bytes}"
        );
        let mut ptr = std::ptr::null_mut();
        // SAFETY: the shim creates and zeroes an allocation owned below.
        check(unsafe { nv_alloc(self.ptr(), bytes, &mut ptr) })?;
        Ok(Buffer {
            allocation: Rc::new(Allocation {
                context: self.context.clone(),
                bytes,
                ptr: NonNull::new(ptr).ok_or_else(|| anyhow!("CUDA returned a null allocation"))?,
            }),
            offset: 0,
            bytes,
        })
    }
    pub fn sync(&self) -> Result<()> {
        self.context.outside_capture()?;
        // SAFETY: live Context owns its handle.
        check(unsafe { nv_sync(self.ptr()) })
    }
    /// Returns (free bytes, total bytes) for this device.
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        self.context.outside_capture()?;
        let (mut free, mut total) = (0, 0);
        // SAFETY: writable size_t output pointers.
        check(unsafe { nv_mem_info(self.ptr(), &mut free, &mut total) })?;
        Ok((free, total))
    }
    pub(crate) fn sm_count(&self) -> Result<usize> {
        let mut sms = 0;
        // SAFETY: live context and writable int output.
        check(unsafe { nv_sm_count(self.ptr(), &mut sms) })?;
        ensure!(sms > 0, "invalid CUDA multiprocessor count");
        Ok(sms as usize)
    }
    pub(crate) fn flash_num_splits(
        &self,
        batch: usize,
        qheads: usize,
        kvheads: usize,
        max_k: usize,
        sms: usize,
    ) -> Result<usize> {
        let args = [batch, qheads, kvheads, max_k, sms].map(dimension);
        let [b, q, k, len, sm] = args;
        let args = (b?, q?, k?, len?, sm?);
        #[cfg(feature = "flash-attn")]
        {
            // SAFETY: pure host heuristic, no pointers or allocation.
            let splits = unsafe { nvr_flash_num_splits(args.0, args.1, args.2, args.3, args.4) };
            ensure!(
                (1..=128).contains(&splits),
                "invalid FlashAttention split count"
            );
            Ok(splits as usize)
        }
        #[cfg(not(feature = "flash-attn"))]
        {
            let _ = args;
            bail!("flash backend requires the flash-attn feature")
        }
    }
    fn tensor(&self, buffer: &Buffer, dims: &[usize], element_size: usize) -> Result<()> {
        ensure!(
            Rc::ptr_eq(&self.context, &buffer.allocation.context),
            "buffer belongs to a different CUDA context"
        );
        buffer.bounds(extent(dims, element_size)?)?;
        ensure!(
            buffer.offset.is_multiple_of(element_size),
            "tensor view is not aligned to its element size"
        );
        if self.context.capturing.get() {
            let mut captured = self.context.captured.borrow_mut();
            if !captured
                .iter()
                .any(|x| Rc::ptr_eq(&x.allocation, &buffer.allocation))
            {
                captured.push(buffer.clone());
            }
        }
        Ok(())
    }
    fn distinct(output: &Buffer, inputs: &[&Buffer]) -> Result<()> {
        ensure!(
            inputs.iter().all(|input| !output.overlaps(input)),
            "kernel output aliases an input or another output"
        );
        Ok(())
    }
    /// Captures only device operations. Captured buffers remain alive with Graph.
    pub fn capture(&self, forward: impl FnOnce() -> Result<()>) -> Result<Graph> {
        self.context.outside_capture()?;
        // SAFETY: no capture active on this stream.
        check(unsafe { nv_capture_begin(self.ptr()) })?;
        self.context.capturing.set(true);
        let mut guard = CaptureGuard {
            context: self.context.clone(),
            active: true,
        };
        forward()?;
        let mut graph = std::ptr::null_mut();
        // nv_capture_end consumes the active capture, including on failure.
        let result = check(unsafe { nv_capture_end(self.ptr(), &mut graph) });
        guard.active = false;
        self.context.capturing.set(false);
        let buffers = std::mem::take(&mut *self.context.captured.borrow_mut());
        result?;
        Ok(Graph {
            context: self.context.clone(),
            handle: NonNull::new(graph).ok_or_else(|| anyhow!("null CUDA graph"))?,
            _buffers: buffers,
        })
    }
    /// X[M,K] times W[N,K]^T, with BF16 inputs/output and FP32 accumulation.
    pub fn gemm(
        &self,
        x: &Buffer,
        w: &Buffer,
        y: &Buffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        self.tensor(x, &[m, k], 2)?;
        self.tensor(w, &[n, k], 2)?;
        self.tensor(y, &[m, n], 2)?;
        Self::distinct(y, &[x, w])?;
        // SAFETY: all shapes, contexts, output aliases and extents checked above.
        check(unsafe {
            nv_gemm(
                self.ptr(),
                x.ptr(),
                w.ptr(),
                y.ptr(),
                dimension(m)?,
                dimension(n)?,
                dimension(k)?,
            )
        })
    }
    pub fn embedding(
        &self,
        tokens: &Buffer,
        weight: &Buffer,
        out: &Buffer,
        rows: usize,
        hidden: usize,
        vocab: usize,
    ) -> Result<()> {
        self.tensor(tokens, &[rows], 4)?;
        self.tensor(weight, &[vocab, hidden], 2)?;
        self.tensor(out, &[rows, hidden], 2)?;
        Self::distinct(out, &[tokens, weight])?;
        check(unsafe {
            nv_embedding(
                self.ptr(),
                tokens.ptr(),
                weight.ptr(),
                out.ptr(),
                dimension(rows)?,
                dimension(hidden)?,
                dimension(vocab)?,
            )
        })
    }
    pub fn rms_norm(
        &self,
        x: &Buffer,
        weight: &Buffer,
        out: &Buffer,
        rows: usize,
        hidden: usize,
        eps: f32,
    ) -> Result<()> {
        positive(eps, "RMS epsilon")?;
        self.tensor(x, &[rows, hidden], 2)?;
        self.tensor(weight, &[hidden], 2)?;
        self.tensor(out, &[rows, hidden], 2)?;
        Self::distinct(out, &[x, weight])?;
        check(unsafe {
            nv_rms(
                self.ptr(),
                x.ptr(),
                weight.ptr(),
                out.ptr(),
                dimension(rows)?,
                dimension(hidden)?,
                eps,
            )
        })
    }
    /// residual <- BF16(x + residual); out <- RMSNorm(float(x)+float(residual)).
    pub fn add_rms_norm(
        &self,
        x: &Buffer,
        residual: &Buffer,
        weight: &Buffer,
        out: &Buffer,
        rows: usize,
        hidden: usize,
        eps: f32,
    ) -> Result<()> {
        positive(eps, "RMS epsilon")?;
        self.tensor(x, &[rows, hidden], 2)?;
        self.tensor(residual, &[rows, hidden], 2)?;
        self.tensor(weight, &[hidden], 2)?;
        self.tensor(out, &[rows, hidden], 2)?;
        Self::distinct(out, &[x, residual, weight])?;
        Self::distinct(residual, &[x, weight])?;
        check(unsafe {
            nv_add_rms(
                self.ptr(),
                x.ptr(),
                residual.ptr(),
                weight.ptr(),
                out.ptr(),
                dimension(rows)?,
                dimension(hidden)?,
                eps,
            )
        })
    }
    fn heads(qheads: usize, kvheads: usize, head_dim: usize) -> Result<()> {
        dimension(qheads)?;
        dimension(kvheads)?;
        dimension(head_dim)?;
        ensure!(
            qheads.is_multiple_of(kvheads),
            "query heads must be a multiple of KV heads"
        );
        ensure!(
            head_dim.is_multiple_of(2) && head_dim <= 256,
            "head dimension must be even and at most 256"
        );
        ensure!(
            qheads.checked_add(kvheads).is_some_and(|n| n <= 65535),
            "too many heads for CUDA grid"
        );
        Ok(())
    }
    pub fn qkv_rope(
        &self,
        qkv: &Buffer,
        qweight: &Buffer,
        kweight: &Buffer,
        positions: &Buffer,
        slots: &Buffer,
        qout: &Buffer,
        kcache: &Buffer,
        vcache: &Buffer,
        rows: usize,
        qheads: usize,
        kvheads: usize,
        head_dim: usize,
        block_size: usize,
        num_blocks: usize,
        eps: f32,
        theta: f32,
    ) -> Result<()> {
        positive(theta, "RoPE theta")?;
        self.check_qkv_rope(
            qkv, qweight, kweight, positions, slots, qout, kcache, vcache, rows, qheads, kvheads,
            head_dim, block_size, num_blocks, eps,
        )?;
        check(unsafe {
            nv_qkv_rope(
                self.ptr(),
                qkv.ptr(),
                qweight.ptr(),
                kweight.ptr(),
                positions.ptr(),
                slots.ptr(),
                qout.ptr(),
                kcache.ptr(),
                vcache.ptr(),
                dimension(rows)?,
                dimension(qheads)?,
                dimension(kvheads)?,
                dimension(head_dim)?,
                dimension(block_size)?,
                dimension(num_blocks)?,
                eps,
                theta,
            )
        })
    }
    fn check_qkv_rope(
        &self,
        qkv: &Buffer,
        qweight: &Buffer,
        kweight: &Buffer,
        positions: &Buffer,
        slots: &Buffer,
        qout: &Buffer,
        kcache: &Buffer,
        vcache: &Buffer,
        rows: usize,
        qheads: usize,
        kvheads: usize,
        head_dim: usize,
        block_size: usize,
        num_blocks: usize,
        eps: f32,
    ) -> Result<()> {
        Self::heads(qheads, kvheads, head_dim)?;
        positive(eps, "RMS epsilon")?;
        let packed_heads = qheads
            .checked_add(
                kvheads
                    .checked_mul(2)
                    .ok_or_else(|| anyhow!("head count overflow"))?,
            )
            .ok_or_else(|| anyhow!("head count overflow"))?;
        self.tensor(qkv, &[rows, packed_heads, head_dim], 2)?;
        for w in [qweight, kweight] {
            self.tensor(w, &[head_dim], 2)?;
        }
        for meta in [positions, slots] {
            self.tensor(meta, &[rows], 4)?;
        }
        self.tensor(qout, &[rows, qheads, head_dim], 2)?;
        for cache in [kcache, vcache] {
            self.tensor(cache, &[num_blocks, block_size, kvheads, head_dim], 2)?;
        }
        let inputs = [qkv, qweight, kweight, positions, slots];
        for out in [qout, kcache, vcache] {
            Self::distinct(out, &inputs)?;
        }
        Self::distinct(qout, &[kcache, vcache])?;
        Self::distinct(kcache, &[vcache])?;
        Ok(())
    }
    /// Build FP32 [positions, cos[head_dim/2] + sin[head_dim/2]] once at load time.
    pub fn rope_cache(
        &self,
        cache: &Buffer,
        max_positions: usize,
        head_dim: usize,
        theta: f32,
    ) -> Result<()> {
        Self::heads(1, 1, head_dim)?;
        positive(theta, "RoPE theta")?;
        self.tensor(cache, &[max_positions, head_dim], 4)?;
        check(unsafe {
            nv_rope_cache(
                self.ptr(),
                cache.ptr(),
                dimension(max_positions)?,
                dimension(head_dim)?,
                theta,
            )
        })
    }
    /// Q/K normalization and RoPE with a persistent FP32 trigonometric table.
    pub fn qkv_rope_cached(
        &self,
        qkv: &Buffer,
        qweight: &Buffer,
        kweight: &Buffer,
        positions: &Buffer,
        slots: &Buffer,
        qout: &Buffer,
        kcache: &Buffer,
        vcache: &Buffer,
        rows: usize,
        qheads: usize,
        kvheads: usize,
        head_dim: usize,
        block_size: usize,
        num_blocks: usize,
        eps: f32,
        rope_cache: &Buffer,
        max_positions: usize,
    ) -> Result<()> {
        self.check_qkv_rope(
            qkv, qweight, kweight, positions, slots, qout, kcache, vcache, rows, qheads, kvheads,
            head_dim, block_size, num_blocks, eps,
        )?;
        self.tensor(rope_cache, &[max_positions, head_dim], 4)?;
        for output in [qout, kcache, vcache] {
            Self::distinct(output, &[rope_cache])?;
        }
        check(unsafe {
            nv_qkv_rope_cached(
                self.ptr(),
                qkv.ptr(),
                qweight.ptr(),
                kweight.ptr(),
                positions.ptr(),
                slots.ptr(),
                qout.ptr(),
                kcache.ptr(),
                vcache.ptr(),
                dimension(rows)?,
                dimension(qheads)?,
                dimension(kvheads)?,
                dimension(head_dim)?,
                dimension(block_size)?,
                dimension(num_blocks)?,
                eps,
                rope_cache.ptr(),
                dimension(max_positions)?,
            )
        })
    }
    /// Causal paged GQA, covering flattened ragged prefill and one-token decode.
    /// Metadata uses i32. KV layout is [blocks, block_size, kvheads, head_dim].
    /// max_context reserves shared memory; each row reads positions[row]+1 keys.
    pub fn paged_attention(
        &self,
        q: &Buffer,
        kcache: &Buffer,
        vcache: &Buffer,
        positions: &Buffer,
        seq_indices: &Buffer,
        block_tables: &Buffer,
        out: &Buffer,
        rows: usize,
        qheads: usize,
        kvheads: usize,
        head_dim: usize,
        block_size: usize,
        num_blocks: usize,
        table_stride: usize,
        batch_size: usize,
        max_context: usize,
    ) -> Result<()> {
        Self::heads(qheads, kvheads, head_dim)?;
        ensure!(
            max_context > 0 && max_context <= 4096,
            "paged attention supports at most 4096 context tokens"
        );
        self.tensor(q, &[rows, qheads, head_dim], 2)?;
        self.tensor(out, &[rows, qheads, head_dim], 2)?;
        for cache in [kcache, vcache] {
            self.tensor(cache, &[num_blocks, block_size, kvheads, head_dim], 2)?;
        }
        for meta in [positions, seq_indices] {
            self.tensor(meta, &[rows], 4)?;
        }
        self.tensor(block_tables, &[batch_size, table_stride], 4)?;
        Self::distinct(
            out,
            &[q, kcache, vcache, positions, seq_indices, block_tables],
        )?;
        check(unsafe {
            nv_attention(
                self.ptr(),
                q.ptr(),
                kcache.ptr(),
                vcache.ptr(),
                positions.ptr(),
                seq_indices.ptr(),
                block_tables.ptr(),
                out.ptr(),
                dimension(rows)?,
                dimension(qheads)?,
                dimension(kvheads)?,
                dimension(head_dim)?,
                dimension(block_size)?,
                dimension(num_blocks)?,
                dimension(table_stride)?,
                dimension(batch_size)?,
                dimension(max_context)?,
            )
        })
    }
    /// Internal API: QwenModel validates grouped query suffixes and physical
    /// page IDs on the host. Upstream kernels trust those device metadata values.
    pub(crate) fn flash_attention(
        &self,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        out: &Buffer,
        cu_q: &Buffer,
        kv_lens: &Buffer,
        tables: &Buffer,
        rows: usize,
        batch: usize,
        max_q: usize,
        max_k: usize,
        qheads: usize,
        kvheads: usize,
        block_size: usize,
        blocks: usize,
        stride: usize,
        splits: usize,
        lse: &Buffer,
        lse_accum: &Buffer,
        out_accum: &Buffer,
    ) -> Result<()> {
        Self::heads(qheads, kvheads, 128)?;
        ensure!(
            qheads <= 256 && block_size > 0 && block_size.is_multiple_of(256),
            "invalid FlashAttention head or block dimensions"
        );
        ensure!(
            max_q > 0
                && max_q <= rows
                && max_k >= max_q
                && max_k <= 4096
                && (1..=128).contains(&splits)
                && (max_q == 1 || splits == 1)
                && (max_q != 1 || rows == batch),
            "invalid FlashAttention launch dimensions"
        );
        for value in [q, out] {
            self.tensor(value, &[rows, qheads, 128], 2)?;
        }
        for value in [k, v] {
            self.tensor(value, &[blocks, block_size, kvheads, 128], 2)?;
        }
        self.tensor(
            cu_q,
            &[batch
                .checked_add(1)
                .ok_or_else(|| anyhow!("batch overflow"))?],
            4,
        )?;
        self.tensor(kv_lens, &[batch], 4)?;
        self.tensor(tables, &[batch, stride], 4)?;
        self.tensor(lse, &[batch, qheads, max_q], 4)?;
        self.tensor(lse_accum, &[splits, batch, qheads], 4)?;
        self.tensor(out_accum, &[splits, batch, qheads, 128], 4)?;
        let inputs = [q, k, v, cu_q, kv_lens, tables];
        let outputs = [out, lse, lse_accum, out_accum];
        for (index, output) in outputs.iter().enumerate() {
            Self::distinct(output, &inputs)?;
            Self::distinct(output, &outputs[..index])?;
        }
        let mut stream = std::ptr::null_mut();
        // SAFETY: activates the owned device; stream lives with this context.
        check(unsafe { nv_stream(self.ptr(), &mut stream) })?;
        #[cfg(feature = "flash-attn")]
        {
            // SAFETY: extents, contexts and aliases checked above; caller has
            // checked page IDs, query grouping, and causal KV lengths. tensor()
            // retains all captured allocations for the graph lifetime.
            let code = unsafe {
                nvr_flash_fwd(
                    stream,
                    q.ptr(),
                    k.ptr(),
                    v.ptr(),
                    out.ptr(),
                    cu_q.ptr(),
                    kv_lens.ptr(),
                    tables.ptr(),
                    dimension(batch)?,
                    dimension(rows)?,
                    dimension(max_q)?,
                    dimension(max_k)?,
                    dimension(qheads)?,
                    dimension(kvheads)?,
                    dimension(block_size)?,
                    dimension(stride)?,
                    dimension(splits)?,
                    lse.ptr(),
                    lse_accum.ptr(),
                    out_accum.ptr(),
                )
            };
            if code != 0 {
                // SAFETY: bridge returns a thread-local, NUL-terminated string.
                let message = unsafe { CStr::from_ptr(nvr_flash_last_error()) }.to_string_lossy();
                bail!("FlashAttention: {message} (status {code})");
            }
            Ok(())
        }
        #[cfg(not(feature = "flash-attn"))]
        {
            bail!("flash backend requires the flash-attn feature")
        }
    }
    pub fn silu_mul(
        &self,
        gateup: &Buffer,
        out: &Buffer,
        rows: usize,
        intermediate: usize,
    ) -> Result<()> {
        self.tensor(gateup, &[rows, 2, intermediate], 2)?;
        self.tensor(out, &[rows, intermediate], 2)?;
        ensure!(
            intermediate <= i32::MAX as usize / 2,
            "intermediate size too large"
        );
        Self::distinct(out, &[gateup])?;
        check(unsafe {
            nv_silu(
                self.ptr(),
                gateup.ptr(),
                out.ptr(),
                dimension(rows)?,
                dimension(intermediate)?,
            )
        })
    }
    pub fn gather(
        &self,
        x: &Buffer,
        indices: &Buffer,
        out: &Buffer,
        rows: usize,
        hidden: usize,
        input_rows: usize,
    ) -> Result<()> {
        self.tensor(x, &[input_rows, hidden], 2)?;
        self.tensor(indices, &[rows], 4)?;
        self.tensor(out, &[rows, hidden], 2)?;
        Self::distinct(out, &[x, indices])?;
        check(unsafe {
            nv_gather(
                self.ptr(),
                x.ptr(),
                indices.ptr(),
                out.ptr(),
                dimension(rows)?,
                dimension(hidden)?,
                dimension(input_rows)?,
            )
        })
    }
    /// BF16 logits; f32 temperatures (zero=greedy); u64 per-row seeds; i32 output.
    /// Positive-temperature sampling uses Gumbel-max with deterministic SplitMix64.
    pub fn sample(
        &self,
        logits: &Buffer,
        temperatures: &Buffer,
        seeds: &Buffer,
        out: &Buffer,
        batch: usize,
        vocab: usize,
    ) -> Result<()> {
        self.tensor(logits, &[batch, vocab], 2)?;
        self.tensor(temperatures, &[batch], 4)?;
        self.tensor(seeds, &[batch], 8)?;
        self.tensor(out, &[batch], 4)?;
        Self::distinct(out, &[logits, temperatures, seeds])?;
        check(unsafe {
            nv_sample(
                self.ptr(),
                logits.ptr(),
                temperatures.ptr(),
                seeds.ptr(),
                out.ptr(),
                dimension(batch)?,
                dimension(vocab)?,
            )
        })
    }
    /// Two-stage vocabulary reduction. Scratch stores one score/id per 2048
    /// vocabulary entries and stays allocated for every captured graph replay.
    pub fn sample_parallel(
        &self,
        logits: &Buffer,
        temperatures: &Buffer,
        seeds: &Buffer,
        out: &Buffer,
        partial_scores: &Buffer,
        partial_ids: &Buffer,
        batch: usize,
        vocab: usize,
    ) -> Result<()> {
        self.tensor(logits, &[batch, vocab], 2)?;
        self.tensor(temperatures, &[batch], 4)?;
        self.tensor(seeds, &[batch], 8)?;
        self.tensor(out, &[batch], 4)?;
        let parts = vocab.div_ceil(2048);
        self.tensor(partial_scores, &[batch, parts], 4)?;
        self.tensor(partial_ids, &[batch, parts], 4)?;
        for output in [out, partial_scores, partial_ids] {
            Self::distinct(output, &[logits, temperatures, seeds])?;
        }
        Self::distinct(out, &[partial_scores, partial_ids])?;
        Self::distinct(partial_scores, &[partial_ids])?;
        check(unsafe {
            nv_sample_parallel(
                self.ptr(),
                logits.ptr(),
                temperatures.ptr(),
                seeds.ptr(),
                out.ptr(),
                partial_scores.ptr(),
                partial_ids.ptr(),
                dimension(batch)?,
                dimension(vocab)?,
            )
        })
    }
}

struct CaptureGuard {
    context: Rc<Context>,
    active: bool,
}
impl Drop for CaptureGuard {
    fn drop(&mut self) {
        if self.active {
            // SAFETY: unwind/error cleanup ends the active capture exactly once.
            unsafe { nv_capture_abort(self.context.ptr()) }
            self.context.capturing.set(false);
            self.context.captured.borrow_mut().clear();
        }
    }
}
/// Executable CUDA graph retaining all captured allocations.
pub struct Graph {
    context: Rc<Context>,
    handle: NonNull<c_void>,
    _buffers: Vec<Buffer>,
}
impl Graph {
    pub fn replay(&self) -> Result<()> {
        self.context.outside_capture()?;
        // SAFETY: graph and all referenced device buffers remain alive.
        check(unsafe { nv_graph_replay(self.context.ptr(), self.handle.as_ptr()) })
    }
}
impl Drop for Graph {
    fn drop(&mut self) {
        // SAFETY: this handle is uniquely owned and context still alive.
        unsafe { nv_graph_destroy(self.context.ptr(), self.handle.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bf16(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|x| {
                let bits = x.to_bits();
                (((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16).to_le_bytes()
            })
            .collect()
    }
    fn values(buffer: &Buffer, count: usize) -> Result<Vec<f32>> {
        Ok(buffer
            .download_bytes(count * 2)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
            .collect())
    }
    #[test]
    #[ignore = "requires an NVIDIA GPU and --features cuda"]
    fn gpu_views_keep_allocations_and_reject_overlap() -> Result<()> {
        let gpu = Gpu::new(0)?;
        let packed = gpu.alloc(32)?;
        packed.upload_i32(&[1, 3, 0, 0, 5, 6, 7, 8])?;
        let tokens = packed.slice(0, 8)?;
        let adjacent = packed.slice(8, 8)?;
        assert!(Gpu::distinct(&tokens, &[&adjacent]).is_ok());
        assert!(Gpu::distinct(&tokens, &[&packed.slice(4, 8)?]).is_err());
        assert!(tokens.slice(4, 8).is_err());
        assert!(packed.slice(usize::MAX, 1).is_err());
        assert!(gpu.tensor(&packed.slice(1, 8)?, &[1], 4).is_err());
        let weights = gpu.alloc(8)?;
        weights.upload_bytes(&bf16(&[1., 2., 3., 4.]))?;
        let out = gpu.alloc(4)?;
        let graph = gpu.capture(|| gpu.embedding(&tokens, &weights, &out, 2, 1, 4))?;
        drop(tokens);
        drop(adjacent);
        drop(packed);
        graph.replay()?;
        assert_eq!(values(&out, 2)?, vec![2., 4.]);
        Ok(())
    }

    #[test]
    #[ignore = "requires an NVIDIA GPU and --features cuda"]
    fn gpu_parallel_sampler_matches_serial_for_tail_ties_and_temperature() -> Result<()> {
        let gpu = Gpu::new(0)?;
        let (batch, vocab) = (3, 151936);
        let mut scores: Vec<f32> = (0..batch * vocab)
            .map(|index| ((index * 31 % 101) as f32 - 50.) * 0.125)
            .collect();
        scores[2047] = 100.;
        scores[2048] = 100.;
        scores[4096] = 100.;
        let logits = gpu.alloc(batch * vocab * 2)?;
        logits.upload_bytes(&bf16(&scores))?;
        let temps = gpu.alloc(batch * 4)?;
        temps.upload_f32(&[0., 0.6, 1.])?;
        let seeds = gpu.alloc(batch * 8)?;
        seeds.upload_u64(&[42, 43, 44])?;
        let serial = gpu.alloc(batch * 4)?;
        let parallel = gpu.alloc(batch * 4)?;
        let partial_scores = gpu.alloc(batch * vocab.div_ceil(2048) * 4)?;
        let partial_ids = gpu.alloc(batch * vocab.div_ceil(2048) * 4)?;
        gpu.sample(&logits, &temps, &seeds, &serial, batch, vocab)?;
        gpu.sample_parallel(
            &logits,
            &temps,
            &seeds,
            &parallel,
            &partial_scores,
            &partial_ids,
            batch,
            vocab,
        )?;
        let expected = serial.download_i32(batch)?;
        assert_eq!(expected[0], 2047);
        assert_eq!(parallel.download_i32(batch)?, expected);
        let graph = gpu.capture(|| {
            gpu.sample_parallel(
                &logits,
                &temps,
                &seeds,
                &parallel,
                &partial_scores,
                &partial_ids,
                batch,
                vocab,
            )
        })?;
        graph.replay()?;
        assert_eq!(parallel.download_i32(batch)?, expected);
        Ok(())
    }

    #[test]
    #[ignore = "requires an NVIDIA GPU and --features cuda"]
    fn gpu_cached_rope_matches_direct_at_context_boundary() -> Result<()> {
        let gpu = Gpu::new(0)?;
        let (rows, qh, kvh, dim) = (4, 4, 2, 128);
        let packed = gpu.alloc(rows * (qh + kvh * 2) * dim * 2)?;
        let input: Vec<f32> = (0..rows * (qh + kvh * 2) * dim)
            .map(|index| ((index * 17 % 37) as f32 - 18.) / 8.)
            .collect();
        packed.upload_bytes(&bf16(&input))?;
        let norm = gpu.alloc(dim * 2)?;
        norm.upload_bytes(&bf16(&vec![1.0; dim]))?;
        let positions = gpu.alloc(rows * 4)?;
        positions.upload_i32(&[0, 1, 350, 4095])?;
        let slots = gpu.alloc(rows * 4)?;
        slots.upload_i32(&[0, 1, 2, 3])?;
        let direct_q = gpu.alloc(rows * qh * dim * 2)?;
        let direct_k = gpu.alloc(4 * kvh * dim * 2)?;
        let direct_v = gpu.alloc(4 * kvh * dim * 2)?;
        let cached_q = gpu.alloc(rows * qh * dim * 2)?;
        let cached_k = gpu.alloc(4 * kvh * dim * 2)?;
        let cached_v = gpu.alloc(4 * kvh * dim * 2)?;
        let cache = gpu.alloc(4096 * dim * 4)?;
        gpu.rope_cache(&cache, 4096, dim, 1e6)?;
        gpu.qkv_rope(
            &packed, &norm, &norm, &positions, &slots, &direct_q, &direct_k, &direct_v, rows, qh,
            kvh, dim, 4, 1, 1e-6, 1e6,
        )?;
        gpu.qkv_rope_cached(
            &packed, &norm, &norm, &positions, &slots, &cached_q, &cached_k, &cached_v, rows, qh,
            kvh, dim, 4, 1, 1e-6, &cache, 4096,
        )?;
        for (direct, cached) in [
            (&direct_q, &cached_q),
            (&direct_k, &cached_k),
            (&direct_v, &cached_v),
        ] {
            assert_eq!(
                direct.download_bytes(direct.len())?,
                cached.download_bytes(cached.len())?
            );
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires an NVIDIA GPU and --features cuda"]
    fn gpu_gemm_rms_sampling_graph() -> Result<()> {
        let gpu = Gpu::new(0)?;
        let x = gpu.alloc(12)?;
        x.upload_bytes(&bf16(&[1., 2., 3., 4., 5., 6.]))?;
        let w = gpu.alloc(12)?;
        w.upload_bytes(&bf16(&[1., 0., 1., 0., 1., 0.]))?;
        let y = gpu.alloc(8)?;
        gpu.gemm(&x, &w, &y, 2, 2, 3)?;
        assert_eq!(values(&y, 4)?, vec![4., 2., 10., 5.]);
        assert!(gpu.gemm(&x, &w, &y, 3, 2, 3).is_err());
        assert!(gpu.gemm(&x, &w, &x, 2, 2, 3).is_err());
        let normw = gpu.alloc(4)?;
        normw.upload_bytes(&bf16(&[1., 1.]))?;
        let normalized = gpu.alloc(8)?;
        gpu.rms_norm(&y, &normw, &normalized, 2, 2, 1e-6)?;
        let norm_values = values(&normalized, 4)?;
        assert!((norm_values[0] - 1.264911).abs() < 0.008);
        let temps = gpu.alloc(8)?;
        temps.upload_f32(&[0., 0.])?;
        let seeds = gpu.alloc(16)?;
        seeds.upload_u64(&[42, 43])?;
        let output = gpu.alloc(8)?;
        gpu.sample(&y, &temps, &seeds, &output, 2, 2)?;
        assert_eq!(output.download_i32(2)?, vec![0, 0]);
        gpu.sync()?;
        let graph = gpu.capture(|| gpu.gemm(&x, &w, &y, 2, 2, 3))?;
        x.upload_bytes(&bf16(&[2., 4., 6., 8., 10., 12.]))?;
        graph.replay()?;
        assert_eq!(values(&y, 4)?, vec![8., 4., 20., 10.]);
        // Captured allocations remain valid even when their original handles drop.
        drop(x);
        drop(w);
        graph.replay()?;
        gpu.sync()?;
        assert!(gpu.capture(|| gpu.alloc(4).map(|_| ())).is_err());
        gpu.sync()?;
        Ok(())
    }
    #[test]
    #[ignore = "requires an NVIDIA GPU and --features cuda"]
    fn gpu_causal_paged_attention_and_rope() -> Result<()> {
        let gpu = Gpu::new(0)?;
        let rows = 3;
        let dim = 32;
        let qh = 2;
        let kvh = 1;
        let q = gpu.alloc(rows * qh * dim * 2)?;
        let k = gpu.alloc(2 * 2 * kvh * dim * 2)?;
        let v = gpu.alloc(2 * 2 * kvh * dim * 2)?;
        // Reversed physical blocks: logical token values 1, 2, 3.
        let mut cache = vec![0.; 4 * dim];
        for d in 0..dim {
            cache[2 * dim + d] = 1.;
            cache[3 * dim + d] = 2.;
            cache[d] = 3.;
        }
        v.upload_bytes(&bf16(&cache))?;
        let positions = gpu.alloc(rows * 4)?;
        positions.upload_i32(&[0, 1, 2])?;
        let seqs = gpu.alloc(rows * 4)?;
        seqs.upload_i32(&[0, 0, 0])?;
        let tables = gpu.alloc(8)?;
        tables.upload_i32(&[1, 0])?;
        let out = gpu.alloc(rows * qh * dim * 2)?;
        gpu.paged_attention(
            &q, &k, &v, &positions, &seqs, &tables, &out, rows, qh, kvh, dim, 2, 2, 2, 1, 4,
        )?;
        let actual = values(&out, rows * qh * dim)?;
        for r in 0..rows {
            for d in 0..qh * dim {
                assert_eq!(actual[r * qh * dim + d], 1. + r as f32 * 0.5);
            }
        }
        let packed = gpu.alloc(rows * (qh + 2 * kvh) * dim * 2)?;
        packed.upload_bytes(&bf16(&vec![1.; rows * (qh + 2 * kvh) * dim]))?;
        let weight = gpu.alloc(dim * 2)?;
        weight.upload_bytes(&bf16(&vec![1.; dim]))?;
        let slots = gpu.alloc(rows * 4)?;
        slots.upload_i32(&[2, 3, 0])?;
        gpu.qkv_rope(
            &packed, &weight, &weight, &positions, &slots, &q, &k, &v, rows, qh, kvh, dim, 2, 2,
            1e-6, 1000000.,
        )?;
        let actualq = values(&q, rows * qh * dim)?;
        assert!(actualq[..qh * dim].iter().all(|x| (*x - 1.).abs() < 0.01));
        assert!((actualq[qh * dim] - (1f32.cos() - 1f32.sin())).abs() < 0.008);
        assert_eq!(values(&v, 4 * dim)?[0], 1.);
        Ok(())
    }
    #[test]
    #[ignore = "requires an NVIDIA GPU and --features cuda"]
    fn gpu_ragged_gqa_matches_cpu() -> Result<()> {
        for (dim, block_size, pos, seq, table) in [
            (32, 2, [0, 2, 1, 0], [0, 0, 1, 2], vec![3, 1, 0, 2, 2, -1]),
            (128, 256, [0, 255, 256, 510], [0, 0, 1, 1], vec![3, 1, 0, 2]),
        ] {
            let gpu = Gpu::new(0)?;
            let (rows, qheads, kvheads, blocks) = (4, 4, 2, 4);
            let query_values: Vec<f32> = (0..rows * qheads * dim)
                .map(|i| ((i * 7 % 19) as f32 - 9.) * 0.125)
                .collect();
            let key_values: Vec<f32> = (0..blocks * block_size * kvheads * dim)
                .map(|i| ((i * 11 % 23) as f32 - 11.) * 0.125)
                .collect();
            let value_values: Vec<f32> = (0..key_values.len())
                .map(|i| ((i * 3 % 13) as f32 - 6.) * 0.125)
                .collect();
            let q = gpu.alloc(query_values.len() * 2)?;
            q.upload_bytes(&bf16(&query_values))?;
            let k = gpu.alloc(key_values.len() * 2)?;
            k.upload_bytes(&bf16(&key_values))?;
            let v = gpu.alloc(value_values.len() * 2)?;
            v.upload_bytes(&bf16(&value_values))?;
            let positions = gpu.alloc(rows * 4)?;
            positions.upload_i32(&pos)?;
            let sequences = gpu.alloc(rows * 4)?;
            sequences.upload_i32(&seq)?;
            let tables = gpu.alloc(table.len() * 4)?;
            tables.upload_i32(&table)?;
            let out = gpu.alloc(query_values.len() * 2)?;
            gpu.paged_attention(
                &q,
                &k,
                &v,
                &positions,
                &sequences,
                &tables,
                &out,
                rows,
                qheads,
                kvheads,
                dim,
                block_size,
                blocks,
                2,
                table.len() / 2,
                *pos.iter().max().unwrap() as usize + 1,
            )?;
            let actual = values(&out, query_values.len())?;
            for row in 0..rows {
                let context = pos[row] as usize + 1;
                for head in 0..qheads {
                    let kvhead = head / (qheads / kvheads);
                    let qbase = (row * qheads + head) * dim;
                    let cache_base = |token: usize| {
                        ((table[seq[row] as usize * 2 + token / block_size] as usize * block_size
                            + token % block_size)
                            * kvheads
                            + kvhead)
                            * dim
                    };
                    let scores: Vec<f32> = (0..context)
                        .map(|token| {
                            (0..dim)
                                .map(|d| {
                                    query_values[qbase + d] * key_values[cache_base(token) + d]
                                })
                                .sum::<f32>()
                                / (dim as f32).sqrt()
                        })
                        .collect();
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let probabilities: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
                    let sum: f32 = probabilities.iter().sum();
                    for d in 0..dim {
                        let expected = (0..context)
                            .map(|t| probabilities[t] * value_values[cache_base(t) + d])
                            .sum::<f32>()
                            / sum;
                        assert!(
                            (actual[qbase + d] - expected).abs() < 0.004,
                            "row {row}, head {head}, dim {d}: {} vs {expected}",
                            actual[qbase + d]
                        );
                    }
                }
            }
        }
        Ok(())
    }
}
