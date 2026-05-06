# neno5/nano5 OpenClaw + SGLang Runtime Setup

This guide reproduces the initial project setup on a Linux GPU server such as `neno5`/`nano5`.

The goal is to:

- Set up OpenClaw.
- Set up SGLang runtime.
- Enable RadixAttention prefix cache.
- Enable KV/request/cache logging.
- Run the R1/R2/R3 prefix-cache benchmark.

## 1. SSH Into The Server

From your local machine:

```bash
ssh <your-user>@<neno5-host>
```

Example, if your server is `nano5.nchc.org.tw`:

```bash
ssh <your-user>@nano5.nchc.org.tw
```

Check GPU access:

```bash
nvidia-smi
```

## 2. Clone The GitHub Repository

On the login node:

```bash
git clone https://github.com/Nyannyan222/openclaw-sglang-prefix-cache.git
cd openclaw-sglang-prefix-cache
```

Run the lightweight check:

```bash
bash scripts/neno5_login_node_check.sh
```

Do not start SGLang on the login node. Use the SLURM job script below.

## 3. Easiest Path: Submit The SLURM Setup + Benchmark Job

From the repository root:

On NCHC nano5/neno5 you may need to pass the project account and a valid `dev`
time limit. For the project shown by SLURM as `MST114180`, use:

```bash
sbatch --account=MST114180 --time=00:30:00 scripts/slurm_setup_and_benchmark.sh
```

Check your job:

```bash
squeue --me
```

After it starts, inspect the output file:

```bash
ls -lh slurm-*-openclaw-sglang.out
tail -n 120 slurm-*-openclaw-sglang.out
```

The job will:

- install a local Node.js runtime under `/work/$USER/openclaw-sglang/node` if
  no cluster Node module is available
- install OpenClaw under `/work/$USER/openclaw-sglang/npm`
- install SGLang under `/work/$USER/openclaw-sglang/runtime/.venv`
- start SGLang on `127.0.0.1:30000`
- enable RadixAttention prefix cache by not passing `--disable-radix-cache`
- enable request/KV cache logging
- configure OpenClaw to use local SGLang
- run the R1/R2/R3 benchmark
- write results under `benchmark_results/neno5_<jobid>/`

If the site requires a different partition or account:

```bash
sbatch -p <partition> -A <account> --time=00:30:00 scripts/slurm_setup_and_benchmark.sh
```

The default script uses:

```text
partition: dev
gpu: 1
time: 30 minutes
memory: 64 GB
```

If downloading Node.js from the compute node is blocked, check cluster modules
on the login node:

```bash
module avail node
```

Then submit with the matching module name:

```bash
sbatch --account=MST114180 --time=00:30:00 --export=ALL,NODE_MODULE=<module-name> scripts/slurm_setup_and_benchmark.sh
```

If the job sees an old compute-node Python such as Python 3.6, check Python
modules on the login node:

```bash
module avail python
```

Then submit with the matching module name:

```bash
sbatch --account=MST114180 --time=00:30:00 --export=ALL,PYTHON_MODULE=<module-name> scripts/slurm_setup_and_benchmark.sh
```

## 4. Manual Setup: Install Basic Tools

Check Python and Node:

```bash
python3 --version
node --version
npm --version
```

If `uv` is missing:

```bash
python3 -m pip install --user uv
```

Make sure your user-local binary path is available:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

## 5. Manual Setup: Install OpenClaw

Install the CLI into your user directory:

```bash
mkdir -p ~/.local
npm install -g openclaw@latest --prefix ~/.local
export PATH="$HOME/.local/bin:$PATH"
openclaw --version
```

Initialize OpenClaw later after SGLang is running, so the SGLang endpoint can be registered as a model provider.

## 6. Manual Setup: Install SGLang Runtime

Create a clean runtime directory:

```bash
mkdir -p ~/openclaw-sglang
cd ~/openclaw-sglang
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install sglang
```

Verify CUDA from PyTorch:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY
```

## 7. Manual Setup: Start SGLang With Prefix Cache And Logging

RadixAttention prefix cache is enabled by default. Do not pass:

```bash
--disable-radix-cache
```

Start SGLang:

```bash
cd ~/openclaw-sglang
source .venv/bin/activate

python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 30000 \
  --enable-metrics \
  --log-requests \
  --log-requests-level 1 \
  --log-requests-format json \
  --radix-eviction-policy lru \
  --mem-fraction-static 0.65
```

If the server has FlashInfer/CUDA graph issues, use the safer baseline mode:

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 30000 \
  --enable-metrics \
  --log-requests \
  --log-requests-level 1 \
  --log-requests-format json \
  --radix-eviction-policy lru \
  --mem-fraction-static 0.65 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --disable-cuda-graph
```

To keep it running in the background:

```bash
cd ~/openclaw-sglang
source .venv/bin/activate

nohup python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 30000 \
  --enable-metrics \
  --log-requests \
  --log-requests-level 1 \
  --log-requests-format json \
  --radix-eviction-policy lru \
  --mem-fraction-static 0.65 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --disable-cuda-graph \
  > /tmp/sglang_openclaw.log 2>&1 &
```

Check the server:

```bash
curl http://127.0.0.1:30000/v1/models
curl http://127.0.0.1:30000/metrics
tail -f /tmp/sglang_openclaw.log
```

In the log, confirm:

```text
disable_radix_cache=False
enable_metrics=True
log_requests=True
KV Cache is allocated
```

## 8. Manual Setup: Configure OpenClaw To Use SGLang

Register the local SGLang OpenAI-compatible endpoint:

```bash
openclaw onboard \
  --non-interactive \
  --accept-risk \
  --mode local \
  --auth-choice custom-api-key \
  --custom-provider-id sglang \
  --custom-base-url http://127.0.0.1:30000/v1 \
  --custom-model-id Qwen/Qwen2.5-0.5B-Instruct \
  --custom-api-key sglang-local \
  --custom-text-input \
  --gateway-auth token \
  --gateway-token local-dev-token \
  --gateway-bind loopback \
  --skip-channels \
  --skip-search \
  --skip-ui \
  --skip-daemon \
  --skip-health
```

Confirm OpenClaw sees the model:

```bash
openclaw models list --provider sglang
openclaw models status
```

Test OpenClaw through SGLang:

```bash
openclaw infer model run \
  --local \
  --model sglang/Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "Reply with exactly: openclaw-sglang-ok" \
  --json
```

Expected output includes:

```json
{
  "ok": true,
  "provider": "sglang",
  "model": "Qwen/Qwen2.5-0.5B-Instruct"
}
```

## 9. Manual Setup: Run The Benchmark

Clone this project repository on `neno5`/`nano5`, then:

```bash
cd <this-repo>
python3 bench_sglang_prefix_cache.py \
  --output-dir benchmark_results \
  --timeout 180 \
  --max-tokens 64
```

The script writes:

```text
benchmark_results/sglang_prefix_cache_<timestamp>.csv
benchmark_results/sglang_prefix_cache_<timestamp>.json
```

The key columns are:

```text
request_name
subcontext_order
usage_prompt_tokens
log_cached_tokens
metric_delta_cached_tokens_total
latency_s
```

Expected pattern:

```text
R1 = A+B+C -> low cached_tokens
R2 = A+B+C -> high cached_tokens
R3 = C+A+B -> much lower cached_tokens than R2
```

## 10. What To Tell The Professor

You can report:

```text
OpenClaw and SGLang runtime are set up.
SGLang RadixAttention prefix cache is enabled by default.
Request/KV cache logging and Prometheus metrics are enabled.
Baseline benchmark shows that repeated ordered prefixes are reused well,
but reordered sub-contexts are not fully reused.
```

This motivates the next stage:

```text
Add sub-context-aware cache instrumentation in SGLang, starting with logging around schedule_policy.match_prefix_for_req().
```
