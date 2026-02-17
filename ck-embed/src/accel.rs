// Hardware acceleration benchmarking and provider selection for ORT-based embedding.
//
// Benchmarks available execution providers (CUDA, TensorRT, OpenVINO, ROCm, CPU)
// and caches results so subsequent runs skip the cost.

use anyhow::{Result, bail};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use ort::execution_providers::{ExecutionProvider, ExecutionProviderDispatch};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Provider list – guarded by target platform cfg, not feature flags.
// The `cpu-only` feature disables every accelerator.
// ---------------------------------------------------------------------------

/// Check whether an execution provider is usable on this system.
///
/// The ORT C-level `GetAvailableProviders` only reports *statically* linked
/// providers.  Providers loaded via external shared libraries (e.g. the
/// OpenVINO EP shipped as `libonnxruntime_providers_openvino.so`) are **not**
/// reported, so `ExecutionProvider::is_available()` returns `false` for them.
///
/// We therefore combine two checks:
///   1. `is_available()` — catches statically-linked providers (CUDA, ROCm …).
///   2. File-based probe — looks for the provider `.so` / `.dll` next to the
///      ORT runtime library we discovered earlier.
fn is_provider_available(name: &str) -> bool {
    // First try the authoritative C-API check (works for static providers).
    let api_available = match name {
        #[cfg(all(not(feature = "cpu-only"), any(
            all(target_os = "linux", target_arch = "x86_64"),
            all(target_os = "linux", target_arch = "aarch64"),
            all(target_os = "windows", target_arch = "x86_64"),
        )))]
        "cuda" => {
            use ort::execution_providers::CUDAExecutionProvider;
            CUDAExecutionProvider::default().is_available().unwrap_or(false)
        }

        #[cfg(all(not(feature = "cpu-only"), any(
            all(target_os = "linux", target_arch = "x86_64"),
            all(target_os = "windows", target_arch = "x86_64"),
        )))]
        "tensorrt" => {
            use ort::execution_providers::TensorRTExecutionProvider;
            TensorRTExecutionProvider::default().is_available().unwrap_or(false)
        }

        #[cfg(all(not(feature = "cpu-only"), any(
            all(target_os = "linux", target_arch = "x86_64"),
            all(target_os = "windows", target_arch = "x86_64"),
        )))]
        "openvino" => {
            use ort::execution_providers::OpenVINOExecutionProvider;
            OpenVINOExecutionProvider::default().is_available().unwrap_or(false)
        }

        #[cfg(all(not(feature = "cpu-only"), all(target_os = "linux", target_arch = "x86_64")))]
        "rocm" => {
            use ort::execution_providers::ROCmExecutionProvider;
            ROCmExecutionProvider::default().is_available().unwrap_or(false)
        }

        #[cfg(all(not(feature = "cpu-only"), all(target_os = "windows", target_arch = "x86_64")))]
        "directml" => {
            use ort::execution_providers::DirectMLExecutionProvider;
            DirectMLExecutionProvider::default().is_available().unwrap_or(false)
        }

        #[cfg(all(not(feature = "cpu-only"), all(target_os = "macos", target_arch = "aarch64")))]
        "coreml" => {
            use ort::execution_providers::CoreMLExecutionProvider;
            CoreMLExecutionProvider::default().is_available().unwrap_or(false)
        }

        "cpu" => return true,
        _ => return false,
    };

    if api_available {
        return true;
    }

    // Fall back to file-based detection: look for the provider shared library
    // next to the ORT dylib we are loading.
    has_provider_shared_lib(name)
}

/// Map a provider name to the expected shared library file that ORT loads at
/// session-creation time.
fn provider_lib_name(name: &str) -> Option<&'static str> {
    match name {
        "openvino" => Some("libonnxruntime_providers_openvino.so"),
        "cuda"     => Some("libonnxruntime_providers_cuda.so"),
        "tensorrt" => Some("libonnxruntime_providers_tensorrt.so"),
        "rocm"     => Some("libonnxruntime_providers_rocm.so"),
        _ => None,
    }
}

/// Check whether the provider shared library exists in one of the discovered
/// ORT runtime directories.
fn has_provider_shared_lib(name: &str) -> bool {
    let lib_name = match provider_lib_name(name) {
        Some(n) => n,
        None => return false,
    };

    // Check env-override first
    if let Ok(dir) = std::env::var("CK_ORT_LIB_DIR") {
        if PathBuf::from(&dir).join(lib_name).is_file() {
            return true;
        }
    }

    // Check all known runtime directories
    let mut dirs = ck_native_runtime_candidates();
    dirs.extend(system_runtime_candidates());
    for dir in dirs {
        if dir.join(lib_name).is_file() {
            return true;
        }
    }

    false
}

/// Returns the list of execution providers that are actually available in the
/// loaded ONNX Runtime library on this system.
pub fn available_providers() -> Result<Vec<String>> {
    // Always ensure the native ORT runtime is discoverable before we query it.
    ensure_ort_runtime_paths();

    let mut providers = vec!["cpu".to_string()];

    #[cfg(not(feature = "cpu-only"))]
    {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        let candidates = ["cuda", "tensorrt", "openvino", "rocm"];

        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        let candidates = ["cuda"];

        #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
        let candidates = ["cuda", "tensorrt", "openvino", "directml"];

        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let candidates = ["coreml"];

        #[cfg(not(any(
            all(target_os = "linux", target_arch = "x86_64"),
            all(target_os = "linux", target_arch = "aarch64"),
            all(target_os = "windows", target_arch = "x86_64"),
            all(target_os = "macos", target_arch = "aarch64"),
        )))]
        let candidates: [&str; 0] = [];

        for name in candidates {
            if is_provider_available(name) {
                // OpenVINO supports multiple device targets; benchmark each one
                // that might be present so the winner is chosen automatically.
                if name == "openvino" {
                    providers.push("openvino-cpu".to_string());
                    providers.push("openvino-gpu".to_string());
                } else {
                    providers.push(name.to_string());
                }
            }
        }
    }

    Ok(providers)
}

// ---------------------------------------------------------------------------
// Benchmark samples – 20 diverse code snippets
// ---------------------------------------------------------------------------

const BENCH_SAMPLES: &[&str] = &[
    "fn main() { println!(\"hello world\"); }",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "class LinkedList<T> { head: Option<Box<Node<T>>> }",
    "SELECT u.name, COUNT(o.id) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.name",
    "impl Iterator for Chunks { type Item = Chunk; fn next(&mut self) -> Option<Self::Item> { self.inner.next() } }",
    "const express = require('express');\nconst app = express();\napp.get('/', (req, res) => res.send('OK'));",
    "async fn fetch_url(url: &str) -> Result<String> { let resp = reqwest::get(url).await?; Ok(resp.text().await?) }",
    "#include <stdio.h>\nint main() { for(int i=0; i<10; i++) printf(\"%d\\n\", i); return 0; }",
    "public record Point(double X, double Y) { public double Distance => Math.Sqrt(X*X + Y*Y); }",
    "func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) { w.Write([]byte(\"OK\")) }",
    "CREATE INDEX idx_users_email ON users(email) WHERE deleted_at IS NULL;",
    "use std::collections::HashMap;\nlet mut map: HashMap<String, Vec<i32>> = HashMap::new();",
    "docker run -d --gpus all -p 8080:8080 -v /data:/data my-model-server:latest",
    "kubectl apply -f deployment.yaml && kubectl rollout status deployment/api-server",
    "interface Props { items: Item[]; onSelect: (id: string) => void; }",
    "from transformers import AutoTokenizer, AutoModel\ntokenizer = AutoTokenizer.from_pretrained('bert-base')",
    "match event { Event::Key(k) => handle_key(k), Event::Mouse(m) => handle_mouse(m), _ => {} }",
    "/// Computes the cosine similarity between two vectors.\npub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x,y)| x*y).sum() }",
    "git log --oneline --graph --all --decorate | head -20",
    "data Tree a = Leaf a | Branch (Tree a) (Tree a) deriving (Show, Eq)",
];

const DEFAULT_BENCH_WORKLOAD_CHUNKS_INDEXING: usize = 240;
const DEFAULT_BENCH_BATCH_SIZE_INDEXING: usize = 64;
const DEFAULT_BENCH_WORKLOAD_CHUNKS_SEARCH: usize = 32;
const DEFAULT_BENCH_BATCH_SIZE_SEARCH: usize = 1;

// ---------------------------------------------------------------------------
// Serializable result types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderResult {
    pub provider: String,
    pub init_ms: f64,
    pub avg_inf_ms: f64,
    pub tokens_per_sec: f64,
    pub workload_time_sec: f64,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkCache {
    #[serde(default)]
    pub version: u32,
    #[serde(default)]
    pub profile: String,
    pub model: String,
    pub timestamp: u64,
    pub system_hash: String,
    pub results: Vec<ProviderResult>,
    pub selected: String,
}

const BENCHMARK_CACHE_VERSION: u32 = 3;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ProviderProfile {
    Search,
    Indexing,
}

impl ProviderProfile {
    fn from_env() -> Self {
        match std::env::var("CK_PROVIDER_PROFILE") {
            Ok(v) if v.eq_ignore_ascii_case("index") || v.eq_ignore_ascii_case("indexing") => {
                Self::Indexing
            }
            _ => Self::Search,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Search => "search",
            Self::Indexing => "indexing",
        }
    }
}

impl BenchmarkCache {
    fn cache_path(model_name: &str, profile: ProviderProfile) -> PathBuf {
        let base = dirs::cache_dir().unwrap_or_else(|| PathBuf::from(".cache"));
        let cache_key = format!("{model_name}:{}", profile.as_str());
        let hash = format!("{:x}", md5::compute(cache_key.as_bytes()));
        base.join("ck")
            .join("benchmarks")
            .join(hash)
            .join("results.json")
    }

    pub fn load(model_name: &str, profile: ProviderProfile) -> Option<Self> {
        let path = Self::cache_path(model_name, profile);
        let data = std::fs::read_to_string(&path).ok()?;
        let cache: Self = serde_json::from_str(&data).ok()?;

        if cache.version != BENCHMARK_CACHE_VERSION {
            eprintln!("[accel] benchmark cache version changed – re-benchmarking");
            return None;
        }

        // Invalidate if system changed
        let current_hash = system_hash();
        if cache.system_hash != current_hash {
            eprintln!("[accel] system changed – re-benchmarking");
            return None;
        }

        // Invalidate if older than 30 days
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        if now.saturating_sub(cache.timestamp) > 30 * 24 * 3600 {
            eprintln!("[accel] cache expired (>30 days) – re-benchmarking");
            return None;
        }

        Some(cache)
    }

    pub fn save(&self, profile: ProviderProfile) -> Result<()> {
        let path = Self::cache_path(&self.model, profile);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// System fingerprinting
// ---------------------------------------------------------------------------

fn run_cmd(program: &str, args: &[&str]) -> Option<String> {
    std::process::Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

fn get_gpu_list() -> String {
    // NVIDIA
    if let Some(gpus) = run_cmd("nvidia-smi", &["--query-gpu=name", "--format=csv,noheader"])
        && !gpus.is_empty()
    {
        return gpus;
    }
    // AMD ROCm
    if let Some(info) = run_cmd("rocm-smi", &["--showproductname"])
        && !info.is_empty()
    {
        return info;
    }
    // Fallback: lspci
    if let Some(pci) = run_cmd("lspci", &[]) {
        let vga: Vec<&str> = pci
            .lines()
            .filter(|l| l.contains("VGA") || l.contains("3D"))
            .collect();
        if !vga.is_empty() {
            return vga.join("\n");
        }
    }
    "none".to_string()
}

fn get_driver_versions() -> String {
    if let Some(v) = run_cmd(
        "nvidia-smi",
        &["--query-gpu=driver_version", "--format=csv,noheader"],
    ) && !v.is_empty()
    {
        return v;
    }
    if let Some(v) = run_cmd("rocm-smi", &["--showdriverversion"])
        && !v.is_empty()
    {
        return v;
    }
    "unknown".to_string()
}

fn system_hash() -> String {
    let fingerprint = format!("{}\n{}", get_gpu_list(), get_driver_versions());
    format!("{:x}", md5::compute(fingerprint.as_bytes()))
}

fn has_file(dir: &std::path::Path, name: &str) -> bool {
    dir.join(name).is_file()
}

fn find_runtime_so(dir: &std::path::Path) -> Option<PathBuf> {
    let canonical = dir.join("libonnxruntime.so");
    if canonical.is_file() {
        return Some(canonical);
    }

    let mut candidates: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("libonnxruntime.so."))
                .unwrap_or(false)
        })
        .collect();

    candidates.sort();
    candidates.pop()
}

fn split_paths(var_name: &str) -> Vec<PathBuf> {
    std::env::var_os(var_name)
        .map(|v| std::env::split_paths(&v).collect())
        .unwrap_or_default()
}

fn ck_native_runtime_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    if let Some(home) = dirs::home_dir() {
        let root = home.join(".cache").join("ck").join("onnxruntime");
        candidates.push(root.join("native-openvino").join("lib"));
        candidates.push(root.join("openvino").join("lib"));

        if let Ok(entries) = std::fs::read_dir(&root) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    candidates.push(path.join("lib"));
                }
            }
        }
    }

    candidates
}

fn system_runtime_candidates() -> Vec<PathBuf> {
    let mut candidates = vec![
        PathBuf::from("/usr/lib"),
        PathBuf::from("/usr/lib64"),
        PathBuf::from("/usr/local/lib"),
        PathBuf::from("/lib"),
        PathBuf::from("/lib64"),
        PathBuf::from("/opt/onnxruntime/lib"),
        PathBuf::from("/opt/intel/onnxruntime/lib"),
        PathBuf::from("/opt/openvino/runtime/lib/intel64"),
    ];

    candidates.extend(split_paths("LD_LIBRARY_PATH"));
    candidates
}

fn discover_ort_runtime_dir(require_openvino_provider: bool) -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("CK_ORT_LIB_DIR") {
        let path = PathBuf::from(dir);
        if has_file(&path, "libonnxruntime_providers_shared.so")
            && (!require_openvino_provider
                || has_file(&path, "libonnxruntime_providers_openvino.so"))
            && find_runtime_so(&path).is_some()
        {
            return Some(path);
        }
    }

    let mut candidates = ck_native_runtime_candidates();
    candidates.extend(system_runtime_candidates());

    for dir in candidates {
        if has_file(&dir, "libonnxruntime_providers_shared.so")
            && (!require_openvino_provider
                || has_file(&dir, "libonnxruntime_providers_openvino.so"))
            && find_runtime_so(&dir).is_some()
        {
            return Some(dir);
        }
    }

    None
}

pub fn discovered_runtime_env() -> Option<(String, String)> {
    let dir = discover_ort_runtime_dir(true)?;
    let runtime_so = find_runtime_so(&dir)?;
    Some((
        runtime_so.to_string_lossy().to_string(),
        dir.to_string_lossy().to_string(),
    ))
}

fn ensure_ort_runtime_paths() {
    if std::env::var("ORT_DYLIB_PATH").is_ok() {
        return;
    }

    let Some(dir) = discover_ort_runtime_dir(true) else {
        return;
    };
    let Some(runtime_so) = find_runtime_so(&dir) else {
        return;
    };

    let runtime_so_str = runtime_so.to_string_lossy().to_string();
    let dir_str = dir.to_string_lossy().to_string();

    unsafe {
        std::env::set_var("ORT_DYLIB_PATH", runtime_so_str);
    }

    if cfg!(target_os = "linux") {
        let merged = match std::env::var("LD_LIBRARY_PATH") {
            Ok(existing) if !existing.is_empty() => format!("{}:{}", dir_str, existing),
            _ => dir_str,
        };
        unsafe {
            std::env::set_var("LD_LIBRARY_PATH", merged);
        }
    }
}

pub fn ensure_runtime_paths() {
    ensure_ort_runtime_paths();
}

fn normalize_provider_error(provider: &str, err: &str) -> String {
    if err.contains("execution provider is not enabled in this build") {
        return format!(
            "{} execution provider is not available in the active ONNX Runtime library bundle",
            provider.to_uppercase()
        );
    }

    if err.contains("Failed to load shared library") {
        return format!(
            "{} provider shared library missing or not loadable (set CK_ORT_LIB_DIR to a valid ONNX Runtime lib directory)",
            provider
        );
    }

    err.to_string()
}

#[derive(Copy, Clone)]
enum WinnerMetric {
    Throughput,
    Inference,
    Workload,
}

fn winner_metric(profile: ProviderProfile) -> WinnerMetric {
    match std::env::var("CK_PROVIDER_SELECTION") {
        Ok(v) if v.eq_ignore_ascii_case("throughput") || v.eq_ignore_ascii_case("tps") => {
            WinnerMetric::Throughput
        }
        Ok(v) if v.eq_ignore_ascii_case("workload") || v.eq_ignore_ascii_case("total") => {
            WinnerMetric::Workload
        }
        Ok(v) if v.eq_ignore_ascii_case("inference") || v.eq_ignore_ascii_case("latency") => {
            WinnerMetric::Inference
        }
        _ => match profile {
            ProviderProfile::Search => WinnerMetric::Inference,
            ProviderProfile::Indexing => WinnerMetric::Throughput,
        },
    }
}

fn benchmark_batch_size(profile: ProviderProfile) -> usize {
    std::env::var("CK_BENCH_BATCH_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(match profile {
            ProviderProfile::Search => DEFAULT_BENCH_BATCH_SIZE_SEARCH,
            ProviderProfile::Indexing => DEFAULT_BENCH_BATCH_SIZE_INDEXING,
        })
}

fn benchmark_workload_chunks(profile: ProviderProfile) -> usize {
    std::env::var("CK_BENCH_WORKLOAD_CHUNKS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(match profile {
            ProviderProfile::Search => DEFAULT_BENCH_WORKLOAD_CHUNKS_SEARCH,
            ProviderProfile::Indexing => DEFAULT_BENCH_WORKLOAD_CHUNKS_INDEXING,
        })
}

fn build_workload_samples(profile: ProviderProfile) -> Vec<String> {
    let target = benchmark_workload_chunks(profile);
    let mut samples = Vec::with_capacity(target);
    for i in 0..target {
        samples.push(BENCH_SAMPLES[i % BENCH_SAMPLES.len()].to_string());
    }
    samples
}

fn run_embed_workload(
    embedding: &mut TextEmbedding,
    samples: &[String],
    batch_size: usize,
) -> Result<(), String> {
    for batch in samples.chunks(batch_size) {
        let batch_vec = batch.to_vec();
        embedding
            .embed(batch_vec, None)
            .map_err(|e| format!("inference error: {e}"))?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Provider construction
// ---------------------------------------------------------------------------

/// Map a provider name string to the corresponding ORT `ExecutionProviderDispatch`.
pub fn build_provider(name: &str) -> Result<ExecutionProviderDispatch> {
    match name {
        #[cfg(all(
            not(feature = "cpu-only"),
            any(
                all(target_os = "linux", target_arch = "x86_64"),
                all(target_os = "linux", target_arch = "aarch64"),
                all(target_os = "windows", target_arch = "x86_64"),
            )
        ))]
        "cuda" => {
            use ort::execution_providers::CUDAExecutionProvider;
            Ok(CUDAExecutionProvider::default().build())
        }

        #[cfg(all(
            not(feature = "cpu-only"),
            any(
                all(target_os = "linux", target_arch = "x86_64"),
                all(target_os = "windows", target_arch = "x86_64"),
            )
        ))]
        "tensorrt" => {
            use ort::execution_providers::TensorRTExecutionProvider;
            Ok(TensorRTExecutionProvider::default().build())
        }

        #[cfg(all(
            not(feature = "cpu-only"),
            any(
                all(target_os = "linux", target_arch = "x86_64"),
                all(target_os = "windows", target_arch = "x86_64"),
            )
        ))]
        "openvino" | "openvino-gpu" | "openvino-cpu" => {
            use ort::execution_providers::OpenVINOExecutionProvider;
            let device = match name {
                "openvino-cpu" => "CPU".to_string(),
                "openvino-gpu" => "GPU".to_string(),
                _ => std::env::var("CK_OPENVINO_DEVICE").unwrap_or_else(|_| "GPU".to_string()),
            };
            let opencl_throttling = std::env::var("CK_OPENVINO_OPENCL_THROTTLING")
                .ok()
                .and_then(|v| match v.to_ascii_lowercase().as_str() {
                    "1" | "true" | "yes" | "on" => Some(true),
                    "0" | "false" | "no" | "off" => Some(false),
                    _ => None,
                })
                .unwrap_or(true);
            Ok(OpenVINOExecutionProvider::default()
                .with_device_type(device)
                .with_opencl_throttling(opencl_throttling)
                .build())
        }

        #[cfg(all(
            not(feature = "cpu-only"),
            all(target_os = "linux", target_arch = "x86_64")
        ))]
        "rocm" => {
            use ort::execution_providers::ROCmExecutionProvider;
            Ok(ROCmExecutionProvider::default().build())
        }

        #[cfg(all(
            not(feature = "cpu-only"),
            all(target_os = "windows", target_arch = "x86_64")
        ))]
        "directml" => {
            use ort::execution_providers::DirectMLExecutionProvider;
            Ok(DirectMLExecutionProvider::default().build())
        }

        #[cfg(all(
            not(feature = "cpu-only"),
            all(target_os = "macos", target_arch = "aarch64")
        ))]
        "coreml" => {
            use ort::execution_providers::CoreMLExecutionProvider;
            Ok(CoreMLExecutionProvider::default().build())
        }

        "cpu" => {
            use ort::execution_providers::CPUExecutionProvider;
            Ok(CPUExecutionProvider::default().build())
        }

        other => bail!("unknown execution provider: {other}"),
    }
}

// ---------------------------------------------------------------------------
// Single-provider benchmark
// ---------------------------------------------------------------------------

pub fn benchmark_one(
    provider: ExecutionProviderDispatch,
    name: &str,
    model: EmbeddingModel,
    profile: ProviderProfile,
) -> ProviderResult {
    let start = Instant::now();

    let cache_dir = match crate::model_cache_root() {
        Ok(d) => d,
        Err(e) => {
            return ProviderResult {
                provider: name.to_string(),
                init_ms: 0.0,
                avg_inf_ms: 0.0,
                tokens_per_sec: 0.0,
                workload_time_sec: 0.0,
                error: Some(format!("cache dir error: {e}")),
            };
        }
    };

    let init_opts = InitOptions::new(model)
        .with_cache_dir(cache_dir)
        .with_execution_providers(vec![provider]);

    let mut embedding = match TextEmbedding::try_new(init_opts) {
        Ok(e) => e,
        Err(e) => {
            return ProviderResult {
                provider: name.to_string(),
                init_ms: start.elapsed().as_secs_f64() * 1000.0,
                avg_inf_ms: 0.0,
                tokens_per_sec: 0.0,
                workload_time_sec: 0.0,
                error: Some(format!("{e}")),
            };
        }
    };
    let init_ms = start.elapsed().as_secs_f64() * 1000.0;

    let samples = build_workload_samples(profile);
    let batch_size = benchmark_batch_size(profile);

    // 2 warmup iterations
    for _ in 0..2 {
        let _ = run_embed_workload(&mut embedding, &samples, batch_size);
    }

    // Measurement iterations: run at least 2 iterations, up to 0.5 seconds total
    let max_measure_time = 0.5; // seconds
    let mut measure_times = Vec::new();
    let mut total_measure = 0.0;
    let mut iterations = 0;
    while iterations < 2 || (iterations < 50 && total_measure < max_measure_time) {
        let iter_start = Instant::now();
        if let Err(e) = run_embed_workload(&mut embedding, &samples, batch_size) {
            return ProviderResult {
                provider: name.to_string(),
                init_ms,
                avg_inf_ms: 0.0,
                tokens_per_sec: 0.0,
                workload_time_sec: 0.0,
                error: Some(e),
            };
        }
        let iter_time = iter_start.elapsed().as_secs_f64();
        measure_times.push(iter_time);
        total_measure += iter_time;
        iterations += 1;
    }

    if iterations == 0 {
        return ProviderResult {
            provider: name.to_string(),
            init_ms,
            avg_inf_ms: 0.0,
            tokens_per_sec: 0.0,
            workload_time_sec: 0.0,
            error: Some("no measurement iterations completed".to_string()),
        };
    }

    let avg_inf_ms = (measure_times.iter().sum::<f64>() / measure_times.len() as f64) * 1000.0;
    let total_tokens_approx = (samples.len() as f64) * 30.0 * (iterations as f64);
    let tokens_per_sec = total_tokens_approx / total_measure;
    let workload_time_sec = init_ms / 1000.0 + total_measure;

    ProviderResult {
        provider: name.to_string(),
        init_ms,
        avg_inf_ms,
        tokens_per_sec,
        workload_time_sec,
        error: None,
    }
}

fn indexing_provider_map() -> &'static Mutex<HashMap<String, String>> {
    static INDEXING_PROVIDER_MAP: OnceLock<Mutex<HashMap<String, String>>> = OnceLock::new();
    INDEXING_PROVIDER_MAP.get_or_init(|| Mutex::new(HashMap::new()))
}

fn remember_indexing_provider(model_name: &str, provider: &str) {
    if let Ok(mut guard) = indexing_provider_map().lock() {
        guard.insert(model_name.to_string(), provider.to_string());
    }
}

fn indexing_provider_for_model(model_name: &str) -> Option<String> {
    let guard = indexing_provider_map().lock().ok()?;
    guard.get(model_name).cloned()
}

// ---------------------------------------------------------------------------
// Main public API
// ---------------------------------------------------------------------------

/// Select the best execution provider for the given model.
///
/// Returns a `Vec<ExecutionProviderDispatch>` suitable for passing to
/// `InitOptions::with_execution_providers`.
pub fn select_provider(
    model: EmbeddingModel,
    model_name: &str,
    force_rebenchmark: bool,
) -> Result<Vec<ExecutionProviderDispatch>> {
    select_provider_with_profile(
        model,
        model_name,
        force_rebenchmark,
        ProviderProfile::from_env(),
    )
}

pub fn select_provider_with_profile(
    model: EmbeddingModel,
    model_name: &str,
    force_rebenchmark: bool,
    profile: ProviderProfile,
) -> Result<Vec<ExecutionProviderDispatch>> {
    ensure_ort_runtime_paths();
    // 1. Check env-var override
    if let Ok(forced) = std::env::var("CK_FORCE_PROVIDER") {
        eprintln!("[accel] CK_FORCE_PROVIDER={forced} – using forced provider");
        let p = build_provider(&forced)?;
        if profile == ProviderProfile::Indexing {
            remember_indexing_provider(model_name, &forced);
        }
        return Ok(vec![p]);
    }

    if profile == ProviderProfile::Search
        && !force_rebenchmark
        && let Some(indexing_provider) = indexing_provider_for_model(model_name)
        && available_providers()?.contains(&indexing_provider)
    {
        eprintln!(
            "[accel] reusing indexing-loaded provider for search: {}",
            indexing_provider
        );
        return Ok(vec![build_provider(&indexing_provider)?]);
    }

    // 2. Check cache
    if !force_rebenchmark {
        if let Some(cache) = BenchmarkCache::load(model_name, profile) {
            let current_available = available_providers()?;
            let cached_providers: std::collections::HashSet<_> = cache.results.iter().map(|r| r.provider.as_str()).collect();
            let current_set: std::collections::HashSet<_> = current_available.iter().map(|s| s.as_str()).collect();
            if current_set.contains(cache.selected.as_str()) && cached_providers.is_subset(&current_set) {
                eprintln!(
                    "[accel] using cached winner: {} (workload {:.2}s)",
                    cache.selected,
                    cache
                        .results
                        .iter()
                        .find(|r| r.provider == cache.selected)
                        .map(|r| r.workload_time_sec)
                        .unwrap_or(0.0)
                );
                let p = build_provider(&cache.selected)?;
                if profile == ProviderProfile::Indexing {
                    remember_indexing_provider(model_name, &cache.selected);
                }
                return Ok(vec![p]);
            } else {
                eprintln!("[accel] cached results outdated (providers changed), re-benchmarking");
            }
        }
    }

    // 3. Benchmark all available providers
    let providers = available_providers()?;
    eprintln!(
        "[accel] benchmarking {} providers: {:?}",
        providers.len(),
        providers
    );

    // Pre-download model to avoid counting download time in first provider's init
    let cache_dir = match crate::model_cache_root() {
        Ok(d) => d,
        Err(e) => bail!("cache dir error: {e}"),
    };
    let pre_init_opts = InitOptions::new(model.clone())
        .with_cache_dir(cache_dir)
        .with_execution_providers(vec![build_provider("cpu")?]);
    let mut _dummy_embedding = TextEmbedding::try_new(pre_init_opts)?;
    let _ = _dummy_embedding.embed(vec!["pre-download test".to_string()], None);
    drop(_dummy_embedding); // Ensure it's dropped before benchmarking

    let mut results: Vec<ProviderResult> = Vec::new();
    for name in &providers {
        eprintln!("[accel]   testing {name}...");
        let provider = match build_provider(name) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[accel]   {name}: build error – {e}");
                results.push(ProviderResult {
                    provider: name.to_string(),
                    init_ms: 0.0,
                    avg_inf_ms: 0.0,
                    tokens_per_sec: 0.0,
                    workload_time_sec: 0.0,
                    error: Some(normalize_provider_error(name, &format!("{e}"))),
                });
                continue;
            }
        };
        let mut result = benchmark_one(provider, name, model.clone(), profile);
        if let Some(err) = result.error.take() {
            result.error = Some(normalize_provider_error(name, &err));
        }
        if let Some(ref err) = result.error {
            eprintln!("[accel]   {name}: error – {err}");
        } else {
            eprintln!(
                "[accel]   {name}: init={:.0}ms  avg_inf={:.1}ms  tok/s={:.0}  total={:.2}s",
                result.init_ms, result.avg_inf_ms, result.tokens_per_sec, result.workload_time_sec
            );
        }
        results.push(result);
    }

    // 4. Pick winner
    let metric = winner_metric(profile);
    let winner = results
        .iter()
        .filter(|r| r.error.is_none())
        .min_by(|a, b| match metric {
            WinnerMetric::Throughput => b
                .tokens_per_sec
                .partial_cmp(&a.tokens_per_sec)
                .unwrap_or(Ordering::Equal)
                .then_with(|| {
                    a.avg_inf_ms
                        .partial_cmp(&b.avg_inf_ms)
                        .unwrap_or(Ordering::Equal)
                }),
            WinnerMetric::Inference => a
                .avg_inf_ms
                .partial_cmp(&b.avg_inf_ms)
                .unwrap_or(Ordering::Equal)
                .then_with(|| {
                    a.workload_time_sec
                        .partial_cmp(&b.workload_time_sec)
                        .unwrap_or(Ordering::Equal)
                }),
            WinnerMetric::Workload => a
                .workload_time_sec
                .partial_cmp(&b.workload_time_sec)
                .unwrap_or(Ordering::Equal)
                .then_with(|| {
                    a.avg_inf_ms
                        .partial_cmp(&b.avg_inf_ms)
                        .unwrap_or(Ordering::Equal)
                }),
        })
        .map(|r| r.provider.clone())
        .unwrap_or_else(|| "cpu".to_string());

    let metric_name = match metric {
        WinnerMetric::Throughput => "throughput",
        WinnerMetric::Inference => "inference",
        WinnerMetric::Workload => "workload",
    };
    eprintln!("[accel] selected provider: {winner} (metric: {metric_name})");

    // 5. Save cache
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let cache = BenchmarkCache {
        version: BENCHMARK_CACHE_VERSION,
        profile: profile.as_str().to_string(),
        model: model_name.to_string(),
        timestamp: now,
        system_hash: system_hash(),
        results,
        selected: winner.clone(),
    };
    if let Err(e) = cache.save(profile) {
        eprintln!("[accel] warning: failed to save cache: {e}");
    }

    if profile == ProviderProfile::Indexing {
        remember_indexing_provider(model_name, &winner);
    }

    // 6. Return
    let p = build_provider(&winner)?;
    Ok(vec![p])
}
