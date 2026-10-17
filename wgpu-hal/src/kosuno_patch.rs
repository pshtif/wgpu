//! Kosuno experiment, not upstream: switchable release of command memory, to
//! measure whether drivers return memory wgpu's command pools keep.
//!
//! Chosen once per process from the Android system property
//! `debug.kosuno.wgpu` (`adb shell setprop debug.kosuno.wgpu release,trim=2`)
//! or the `KOSUNO_WGPU` environment variable elsewhere:
//!
//! - `release`: reset command pools with `VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT`
//!   (Vulkan), so the driver may free their memory instead of keeping it.
//! - `trim=N`: keep at most `N` idle command encoders (each a command pool)
//!   in wgpu-core's pool; drop the rest, destroying their pools.
//! - `age=MS`: drop idle encoders unused for more than `MS` milliseconds.
//! - `release_every=N`: each pool releases its resources on every `N`th reset.
//! - `adaptive`: a pool releases its resources once its recent work (render
//!   passes recorded per reset) has stayed under a quarter of its peak for
//!   60 resets: memory returns after a spike, steady heavy work never pays.
//! - `adaptive2`: the same decided across all pools: when the busiest pool
//!   reset of the last 120 resets did under a quarter of the peak's work,
//!   every pool releases once at its next reset.
//! - `adaptive3`: `adaptive2` with hysteresis in time: release only after
//!   work stayed calm for a delay (5 s), and double the delay (to 60 s)
//!   when heavy work returns within 30 s of a release, so alternating
//!   heavy and light frames keep their memory instead of thrashing.
//! - `adaptive4`: `adaptive3`, but instead of releasing pools (after which
//!   Adreno records commands slower for good), wgpu-core destroys the
//!   encoders acquired before the calm epoch as they come back, and
//!   creates fresh ones.
//! - `adaptive5`: `adaptive3`, but trims pools (`vkTrimCommandPool`) after
//!   their next reset instead of releasing them.
//! - `stats`: log the mean number of command buffers per queue submit.
//!
//! One command buffer per encoder (instead of two per pass) is not a variant
//! here: it is always on, in wgpu-core (`command/pass_recorder.rs`).
//!
//! Unset, the setting is [`DEFAULT`]: what Kosuno ships with. `none` (or any
//! value without known tokens) gives stock wgpu behaviour.

use core::sync::atomic::{AtomicU32, Ordering};

/// Bit 31: parsed. Bit 30: release. Bits 0..16: trim limit + 1 (0 = none).
static MODE: AtomicU32 = AtomicU32::new(0);
static AGE_MS: AtomicU32 = AtomicU32::new(0);
static RELEASE_EVERY: AtomicU32 = AtomicU32::new(0);
static ADAPTIVE: AtomicU32 = AtomicU32::new(0);
static GLOBAL: AtomicU32 = AtomicU32::new(0);
static HYSTERESIS: AtomicU32 = AtomicU32::new(0);
static DESTROY: AtomicU32 = AtomicU32::new(0);
static TRIM_POOL: AtomicU32 = AtomicU32::new(0);
static EPOCH: AtomicU32 = AtomicU32::new(0);
static PEAK: AtomicU32 = AtomicU32::new(0);
static WINDOW_MAX: AtomicU32 = AtomicU32::new(0);
static WINDOW_RESETS: AtomicU32 = AtomicU32::new(0);

/// The variants used when neither the property nor the variable is set.
pub const DEFAULT: &str = "adaptive5";

fn mode() -> u32 {
    let m = MODE.load(Ordering::Relaxed);
    if m & (1 << 31) != 0 {
        return m;
    }
    let mut m = 1u32 << 31;
    if let Some(value) = setting().or_else(|| Some(DEFAULT.into())) {
        for token in value.split(',').map(str::trim) {
            if token == "release" {
                m |= 1 << 30;
            } else if let Some(n) = token.strip_prefix("trim=") {
                if let Ok(n) = n.parse::<u32>() {
                    m |= (n.min(0xFFFE) + 1) & 0xFFFF;
                }
            } else if let Some(n) = token.strip_prefix("age=") {
                AGE_MS.store(n.parse().unwrap_or(0), Ordering::Relaxed);
            } else if let Some(n) = token.strip_prefix("release_every=") {
                RELEASE_EVERY.store(n.parse().unwrap_or(0), Ordering::Relaxed);
            } else if token == "stats" {
                STATS.store(1, Ordering::Relaxed);
            } else if token == "adaptive5" {
                GLOBAL.store(1, Ordering::Relaxed);
                HYSTERESIS.store(1, Ordering::Relaxed);
                TRIM_POOL.store(1, Ordering::Relaxed);
            } else if token == "adaptive4" {
                GLOBAL.store(1, Ordering::Relaxed);
                HYSTERESIS.store(1, Ordering::Relaxed);
                DESTROY.store(1, Ordering::Relaxed);
            } else if token == "adaptive3" {
                GLOBAL.store(1, Ordering::Relaxed);
                HYSTERESIS.store(1, Ordering::Relaxed);
            } else if token == "adaptive2" {
                GLOBAL.store(1, Ordering::Relaxed);
            } else if token == "adaptive" {
                ADAPTIVE.store(1, Ordering::Relaxed);
            } else if token == "trim" {
                m |= 2 + 1;
            }
        }
        log::warn!("kosuno wgpu patch: '{value}' → release={} trim={:?} age_ms={} release_every={}", m & (1 << 30) != 0, trim_of(m),
            AGE_MS.load(Ordering::Relaxed), RELEASE_EVERY.load(Ordering::Relaxed));
        log::warn!("kosuno wgpu patch: adaptive={} adaptive2={}", ADAPTIVE.load(Ordering::Relaxed) != 0, GLOBAL.load(Ordering::Relaxed) != 0);
    }
    MODE.store(m, Ordering::Relaxed);
    m
}

fn trim_of(m: u32) -> Option<usize> {
    let t = m & 0xFFFF;
    (t != 0).then(|| (t - 1) as usize)
}

/// Reset command pools releasing their resources to the system.
pub fn release_pools() -> bool {
    mode() & (1 << 30) != 0
}

/// Drop idle encoders unused for longer than this, in milliseconds.
pub fn age_ms() -> Option<u32> {
    mode();
    let a = AGE_MS.load(Ordering::Relaxed);
    (a != 0).then_some(a)
}

/// Release a pool's resources on every this-many resets.
pub fn release_every() -> Option<u32> {
    mode();
    let n = RELEASE_EVERY.load(Ordering::Relaxed);
    (n != 0).then_some(n)
}

/// Release pools once their work has dropped well below its peak.
pub fn adaptive() -> bool {
    mode();
    ADAPTIVE.load(Ordering::Relaxed) != 0
}

/// Release pools once work across all of them dropped well below the peak.
pub fn adaptive_global() -> bool {
    mode();
    GLOBAL.load(Ordering::Relaxed) != 0
}

/// At a pool reset under [`adaptive_global`], with the render passes it
/// recorded: whether to release its resources now. `epoch` is the pool's.
pub fn global_reset(passes: u32, epoch: &mut u32) -> bool {
    if HYSTERESIS.load(Ordering::Relaxed) != 0 {
        return hysteresis_reset(passes, epoch);
    }
    WINDOW_MAX.fetch_max(passes, Ordering::Relaxed);
    if WINDOW_RESETS.fetch_add(1, Ordering::Relaxed) + 1 >= 120 {
        WINDOW_RESETS.store(0, Ordering::Relaxed);
        let window = WINDOW_MAX.swap(0, Ordering::Relaxed);
        let peak = PEAK.load(Ordering::Relaxed);
        if peak >= 32 && window * 4 < peak {
            // The spike is over: every pool releases once.
            EPOCH.fetch_add(1, Ordering::Relaxed);
            PEAK.store(window, Ordering::Relaxed);
            log::warn!("kosuno wgpu patch: work fell from {peak} to {window} passes per reset; releasing pools");
        } else {
            PEAK.fetch_max(window, Ordering::Relaxed);
        }
    }
    let now = EPOCH.load(Ordering::Relaxed);
    if *epoch != now {
        *epoch = now;
        return now != 0;
    }
    false
}

/// Per-pool state for [`adaptive`]: render passes recorded since the last
/// reset, the peak since the last release, and resets spent well below it.
#[derive(Debug, Default)]
pub struct Adaptive {
    pub passes: u32,
    pub epoch: u32,
    peak: u32,
    calm: u32,
}

impl Adaptive {
    /// At a pool reset: whether to release its resources now.
    pub fn reset(&mut self) -> bool {
        let passes = core::mem::take(&mut self.passes);
        self.peak = self.peak.max(passes);
        if self.peak >= 32 && passes * 4 < self.peak {
            self.calm += 1;
            if self.calm >= 60 {
                self.peak = passes;
                self.calm = 0;
                return true;
            }
        } else {
            self.calm = 0;
        }
        false
    }
}

/// Keep at most this many idle command encoders pooled.
pub fn trim_encoders() -> Option<usize> {
    trim_of(mode())
}

#[cfg(target_os = "android")]
fn setting() -> Option<alloc::string::String> {
    extern "C" {
        fn __system_property_get(name: *const core::ffi::c_char, value: *mut core::ffi::c_char) -> core::ffi::c_int;
    }
    let mut buf = [0u8; 92];
    let len = unsafe { __system_property_get(c"debug.kosuno.wgpu".as_ptr(), buf.as_mut_ptr().cast()) };
    if len <= 0 {
        return None;
    }
    core::str::from_utf8(&buf[..len as usize]).ok().map(alloc::string::String::from)
}

#[cfg(all(not(target_os = "android"), any(dx12, gles_with_std, metal, vulkan)))]
fn setting() -> Option<alloc::string::String> {
    std::env::var("KOSUNO_WGPU").ok()
}

#[cfg(all(not(target_os = "android"), not(any(dx12, gles_with_std, metal, vulkan))))]
fn setting() -> Option<alloc::string::String> {
    None
}

#[cfg(any(dx12, gles_with_std, metal, vulkan))]
struct Hysteresis {
    peak: u32,
    window_max: u32,
    window_resets: u32,
    calm_since: Option<std::time::Instant>,
    delay: core::time::Duration,
    last_release: Option<std::time::Instant>,
}

#[cfg(any(dx12, gles_with_std, metal, vulkan))]
fn hysteresis_reset(passes: u32, epoch: &mut u32) -> bool {
    use core::time::Duration;
    use std::time::Instant;
    static STATE: std::sync::Mutex<Option<Hysteresis>> = std::sync::Mutex::new(None);
    let mut guard = STATE.lock().unwrap();
    let h = guard.get_or_insert(Hysteresis {
        peak: 0, window_max: 0, window_resets: 0, calm_since: None,
        delay: Duration::from_secs(5), last_release: None,
    });
    h.window_max = h.window_max.max(passes);
    h.window_resets += 1;
    if h.window_resets >= 120 {
        let window = core::mem::take(&mut h.window_max);
        h.window_resets = 0;
        let now = Instant::now();
        if h.peak >= 32 && window * 4 < h.peak {
            let since = *h.calm_since.get_or_insert(now);
            if now.duration_since(since) >= h.delay {
                EPOCH.fetch_add(1, Ordering::Relaxed);
                log::warn!("kosuno wgpu patch: calm for {:?} (work {} vs peak {}); releasing pools", h.delay, window, h.peak);
                h.peak = window;
                h.calm_since = None;
                h.last_release = Some(now);
            }
        } else {
            h.calm_since = None;
            // Heavy work back soon after a release: wait longer next time.
            if window >= 32 {
                if let Some(at) = h.last_release.take() {
                    if now.duration_since(at) < Duration::from_secs(30) {
                        h.delay = (h.delay * 2).min(Duration::from_secs(60));
                        log::warn!("kosuno wgpu patch: heavy work returned {:?} after a release; delay now {:?}", now.duration_since(at), h.delay);
                    }
                }
            }
            h.peak = h.peak.max(window);
        }
    }
    drop(guard);
    let now = EPOCH.load(Ordering::Relaxed);
    if *epoch != now {
        *epoch = now;
        // `adaptive4`: wgpu-core replaces the pool instead.
        return now != 0 && DESTROY.load(Ordering::Relaxed) == 0;
    }
    false
}

/// `adaptive5`: trim instead of releasing.
pub fn trim_pool() -> bool {
    mode();
    TRIM_POOL.load(Ordering::Relaxed) != 0
}

/// `adaptive4`: the calm epoch, for wgpu-core to replace encoders acquired
/// before it. `None` when not destroying.
pub fn destroy_epoch() -> Option<u32> {
    mode();
    (DESTROY.load(Ordering::Relaxed) != 0).then(|| EPOCH.load(Ordering::Relaxed))
}

#[cfg(not(any(dx12, gles_with_std, metal, vulkan)))]
fn hysteresis_reset(_: u32, _: &mut u32) -> bool {
    false
}

static SUBMITS: AtomicU32 = AtomicU32::new(0);
static SUBMITTED_BUFFERS: AtomicU32 = AtomicU32::new(0);

/// Counts command buffers per queue submit; logs the mean every 600 submits
/// when `KOSUNO_WGPU` / `debug.kosuno.wgpu` contains `stats`.
pub fn note_submit(buffers: usize) {
    if !stats() {
        return;
    }
    let total = SUBMITTED_BUFFERS.fetch_add(buffers as u32, Ordering::Relaxed) + buffers as u32;
    let n = SUBMITS.fetch_add(1, Ordering::Relaxed) + 1;
    if n >= 600 {
        SUBMITS.store(0, Ordering::Relaxed);
        SUBMITTED_BUFFERS.store(0, Ordering::Relaxed);
        log::warn!("kosuno wgpu patch: {:.1} command buffers per submit", total as f32 / n as f32);
    }
}

static STATS: AtomicU32 = AtomicU32::new(0);

fn stats() -> bool {
    mode();
    STATS.load(Ordering::Relaxed) != 0
}
