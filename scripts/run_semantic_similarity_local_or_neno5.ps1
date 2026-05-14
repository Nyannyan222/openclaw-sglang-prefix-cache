param(
    [string]$SemanticJsonl = "benchmark_results/wildclaw_semantic_subcontext_pilot_v3/wildclaw_semantic_subcontext_pilot.jsonl",
    [string]$OutputDir = "benchmark_results/semantic_similarity_kv_reuse",
    [double]$Threshold = 0.05,
    [int]$MaxPairs = 8,
    [string]$Python = ".\.venv\Scripts\python.exe",
    [switch]$RunRuntime,
    [string]$BaseUrl = "http://127.0.0.1:30000/v1",
    [string]$MetricsUrl = "http://127.0.0.1:30000/metrics",
    [string]$Model = "Qwen/Qwen2.5-0.5B-Instruct",
    [int]$Repeat = 1,
    [int]$Limit = 0,
    [int]$MaxTokens = 64,
    [string]$Account = "MST114180"
)

$ErrorActionPreference = "Stop"

function Resolve-Python {
    param([string]$Candidate)
    if (Test-Path $Candidate) {
        return $Candidate
    }
    return "python"
}

function Test-SglangEndpoint {
    param([string]$OpenAiBaseUrl)
    $modelsUrl = $OpenAiBaseUrl.TrimEnd("/") + "/models"
    try {
        $response = Invoke-WebRequest -Uri $modelsUrl -UseBasicParsing -TimeoutSec 3
        return ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300)
    }
    catch {
        return $false
    }
}

function Write-Neno5Command {
    param(
        [string]$Manifest,
        [string]$Account,
        [int]$Repeat,
        [int]$Limit
    )

    Write-Host ""
    Write-Host "No local SGLang endpoint is ready. Run the runtime replay on neno5:"
    Write-Host ""
    Write-Host "git pull origin main"
    Write-Host ""
    Write-Host "sbatch --account=$Account \"
    Write-Host "  --export=ALL,MANIFEST=$Manifest,LIMIT=$Limit,REPEAT=$Repeat \"
    Write-Host "  scripts/slurm_run_wildclaw_runtime_replay.sh"
}

$Python = Resolve-Python $Python

if (-not (Test-Path $SemanticJsonl)) {
    throw "Semantic JSONL not found: $SemanticJsonl"
}

Write-Host "== Local stage: prepare semantic similarity manifest =="
& $Python scripts/prepare_semantic_similarity_kv_reuse.py `
    --semantic-jsonl $SemanticJsonl `
    --output-dir $OutputDir `
    --threshold $Threshold `
    --max-pairs $MaxPairs

$manifest = (Join-Path $OutputDir "semantic_similarity_runtime_manifest.jsonl").Replace("\", "/")
if (-not (Test-Path $manifest)) {
    throw "Runtime manifest was not produced: $manifest"
}

Write-Host ""
Write-Host "Manifest ready: $manifest"

if (-not $RunRuntime) {
    Write-Host ""
    Write-Host "Runtime replay was not requested. To try local runtime, start SGLang on $BaseUrl and rerun with -RunRuntime."
    Write-Neno5Command -Manifest $manifest -Account $Account -Repeat $Repeat -Limit $Limit
    exit 0
}

Write-Host ""
Write-Host "== Local stage: check SGLang endpoint =="
if (-not (Test-SglangEndpoint $BaseUrl)) {
    Write-Neno5Command -Manifest $manifest -Account $Account -Repeat $Repeat -Limit $Limit
    exit 0
}

Write-Host "Local SGLang endpoint is ready: $BaseUrl"
Write-Host ""
Write-Host "== Local stage: run runtime replay =="

$replayArgs = @(
    "scripts/run_wildclaw_sglang_runtime_replay.py",
    "--manifest", $manifest,
    "--base-url", $BaseUrl,
    "--metrics-url", $MetricsUrl,
    "--model", $Model,
    "--repeat", "$Repeat",
    "--max-tokens", "$MaxTokens"
)

if ($Limit -ne 0) {
    $replayArgs += @("--limit", "$Limit")
}

& $Python @replayArgs

$runRoot = "benchmark_results/wildclaw_sglang_runtime_runs"
$latestRun = Get-ChildItem $runRoot -Directory |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($null -eq $latestRun) {
    throw "No runtime run directory found under $runRoot"
}

$csvPath = Join-Path $latestRun.FullName "wildclaw_sglang_runtime_results.csv"
$summaryPath = Join-Path $latestRun.FullName "wildclaw_sglang_runtime_summary.md"

& $Python scripts/summarize_wildclaw_runtime_results.py $csvPath --output $summaryPath

Write-Host ""
Write-Host "Runtime summary:"
Write-Host $summaryPath
Get-Content $summaryPath
