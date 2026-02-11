// Hardware acceleration benchmarking and provider selection for ORT-based embedding.
//
// Benchmarks available execution providers (CUDA, TensorRT, OpenVINO, ROCm, CPU)
// and caches results so subsequent runs skip the cost.

use anyhow::{Result, bail};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use ort::execution_providers::ExecutionProviderDispatch;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Provider list – guarded by target platform cfg, not feature flags.
// The `cpu-only` feature disables every accelerator.
// ---------------------------------------------------------------------------

/// Returns the list of execution providers that are available on this system.
pub fn available_providers() -> Result<Vec<String>> {
    let mut providers = vec!["cpu".to_string()];

    #[cfg(not(feature = "cpu-only"))]
    {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            let candidates = ["cuda", "tensorrt", "openvino", "rocm"];
            for name in candidates {
                if build_provider(name).is_ok() {
                    providers.push(name.to_string());
                }
            }
        }

        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        {
            if build_provider("cuda").is_ok() {
                providers.push("cuda".to_string());
            }
        }

        #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
        {
            let candidates = ["cuda", "tensorrt", "openvino", "directml"];
            for name in candidates {
                if build_provider(name).is_ok() {
                    providers.push(name.to_string());
                }
            }
        }

        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            if build_provider("coreml").is_ok() {
                providers.push("coreml".to_string());
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
    pub model: String,
    pub timestamp: u64,
    pub system_hash: String,
    pub results: Vec<ProviderResult>,
    pub selected: String,
}

impl BenchmarkCache {
    fn cache_path(model_name: &str) -> PathBuf {
        let base = dirs::cache_dir().unwrap_or_else(|| PathBuf::from(".cache"));
        let hash = format!("{:x}", md5::compute(model_name.as_bytes()));
        base.join("ck")
            .join("benchmarks")
            .join(hash)
            .join("results.json")
    }

    pub fn load(model_name: &str) -> Option<Self> {
        let path = Self::cache_path(model_name);
        let data = std::fs::read_to_string(&path).ok()?;
        let cache: Self = serde_json::from_str(&data).ok()?;

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

    pub fn save(&self) -> Result<()> {
        let path = Self::cache_path(&self.model);
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
        "openvino" => {
            use ort::execution_providers::OpenVINOExecutionProvider;
            Ok(OpenVINOExecutionProvider::default().build())
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

    let samples: Vec<String> = BENCH_SAMPLES.iter().map(|s| s.to_string()).collect();

    // 2 warmup iterations
    for _ in 0..2 {
        let _ = embedding.embed(samples.clone(), None);
    }

    // Measurement iterations: run at least 2 iterations, up to 0.5 seconds total
    let max_measure_time = 0.5; // seconds
    let mut measure_times = Vec::new();
    let mut total_measure = 0.0;
    let mut iterations = 0;
    while iterations < 2 || (iterations < 50 && total_measure < max_measure_time) {
        let iter_start = Instant::now();
        if let Err(e) = embedding.embed(samples.clone(), None) {
            return ProviderResult {
                provider: name.to_string(),
                init_ms,
                avg_inf_ms: 0.0,
                tokens_per_sec: 0.0,
                workload_time_sec: 0.0,
                error: Some(format!("inference error: {e}")),
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
    let total_tokens_approx = (BENCH_SAMPLES.len() as f64) * 30.0 * (iterations as f64);
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
    // 1. Check env-var override
    if let Ok(forced) = std::env::var("CK_FORCE_PROVIDER") {
        eprintln!("[accel] CK_FORCE_PROVIDER={forced} – using forced provider");
        let p = build_provider(&forced)?;
        return Ok(vec![p]);
    }

    // 2. Check cache
    if !force_rebenchmark {
        if let Some(cache) = BenchmarkCache::load(model_name) {
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
                    error: Some(format!("{e}")),
                });
                continue;
            }
        };
        let result = benchmark_one(provider, name, model.clone());
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

    // 4. Pick the winner – lowest workload_time_sec among successes
    let winner = results
        .iter()
        .filter(|r| r.error.is_none())
        .min_by(|a, b| {
            a.workload_time_sec
                .partial_cmp(&b.workload_time_sec)
                .unwrap()
        })
        .map(|r| r.provider.clone())
        .unwrap_or_else(|| "cpu".to_string());

    eprintln!("[accel] selected provider: {winner}");

    // 5. Save cache
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let cache = BenchmarkCache {
        model: model_name.to_string(),
        timestamp: now,
        system_hash: system_hash(),
        results,
        selected: winner.clone(),
    };
    if let Err(e) = cache.save() {
        eprintln!("[accel] warning: failed to save cache: {e}");
    }

    // 6. Return
    let p = build_provider(&winner)?;
    Ok(vec![p])
}
