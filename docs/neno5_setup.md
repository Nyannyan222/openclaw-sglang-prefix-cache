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

## 2. Install Basic Tools

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

## 3. Install OpenClaw

Install the CLI into your user directory:

```bash
mkdir -p ~/.local
npm install -g openclaw@latest --prefix ~/.local
export PATH="$HOME/.local/bin:$PATH"
openclaw --version
```

Initialize OpenClaw later after SGLang is running, so the SGLang endpoint can be registered as a model provider.

## 4. Install SGLang Runtime

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

## 5. Start SGLang With Prefix Cache And Logging

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

## 6. Configure OpenClaw To Use SGLang

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

## 7. Run The Benchmark

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

## 8. What To Tell The Professor

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

