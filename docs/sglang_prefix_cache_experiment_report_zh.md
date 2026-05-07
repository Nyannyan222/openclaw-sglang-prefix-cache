# SGLang RadixAttention Prefix Cache 與 Sub-Context Cache 實驗報告

## 摘要

本研究目標是觀察 SGLang RadixAttention prefix cache 在重複上下文與重新排序上下文中的行為，並進一步設計 sub-context-aware cache prototype，評估它是否能補足原生 prefix cache 對順序敏感的限制。

目前已完成三個階段：

1. 在 neno5 / WSL 建立 SGLang runtime，啟用 RadixAttention prefix cache、metrics 與 request/cache logging。
2. 以 R1/R2/R3 驗證原生 RadixAttention prefix cache：相同 prefix 可達約 96% cached token ratio，但重新排序後下降至約 19%。
3. 設計 `SubContextIndex` prototype，並完成 No cache、SGLang RadixAttention、Sub-context-aware cache 三組對照實驗。

核心結論是：SGLang 原生 RadixAttention prefix cache 對完整 prefix 重複非常有效，但對 `A+B+C` 改成 `C+A+B`、`B+C+A`、`A+C+B` 這類 reordered sub-context 無法完整重用。外掛式 `SubContextIndex` prototype 可以根據 sub-context hash 辨識已出現過的 A/B/C 片段，顯示 segment-level reuse 的潛在空間。

需要注意的是，目前 Proposed 方法仍是 metadata-level prototype，尚未真正 splice 或重用 non-prefix KV blocks。因此 token reuse / prefill token 可以估算，但 latency improvement 還不能宣稱已被實測。

## 1. 實驗目的

本實驗根據教授給定方向，先完成 OpenClaw 與 SGLang runtime 初始環境建置，並啟用：

- SGLang RadixAttention prefix cache
- SGLang metrics endpoint
- request logging
- KV/cache lookup logging
- sub-context metadata export

主要研究問題如下：

1. SGLang RadixAttention prefix cache 在相同 prefix 下是否能有效命中？
2. 當相同 sub-context 重新排序時，prefix cache 是否仍能重用？
3. 不同模型大小是否會改變 prefix cache 對順序的敏感性？
4. 若在 RadixAttention prefix cache 外增加一層 `SubContextIndex`，是否能辨識 reordered sub-context 的 reuse opportunity？
5. No cache、原生 RadixAttention、Sub-context-aware prototype 三組方法在 cached token ratio、prefill tokens、first token latency、total latency 上有何差異？

## 2. 實驗環境

主要 neno5 實驗於 GPU compute node 上透過 SLURM job 執行。另有 WSL 本機小模型實驗，用於快速驗證 logging、prototype 與三組 baseline comparison。

### 2.1 neno5 環境

| 項目 | 設定 |
|---|---|
| 平台 | neno5 / SLURM |
| GPU | NVIDIA H100 80GB HBM3 |
| CUDA module | cuda/12.4 |
| Compiler module | gcc/12.5 |
| Runtime | SGLang |
| SGLang version | sglang==0.5.9 |
| OpenClaw | 2026.5.5 |
| Python | 3.11 |
| Prefix cache | RadixAttention prefix cache enabled |
| Logging | SGLang request logging / metrics / cache lookup logging |

SGLang server 主要啟動參數如下：

```bash
python -m sglang.launch_server \
  --model-path "$MODEL_ID" \
  --host 127.0.0.1 \
  --port "$SGLANG_PORT" \
  --enable-metrics \
  --log-requests \
  --log-requests-level 1 \
  --log-requests-format json \
  --radix-eviction-policy lru \
  --mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC" \
  --attention-backend triton \
  --sampling-backend pytorch \
  --disable-cuda-graph
```

### 2.2 WSL 驗證環境

| 項目 | 設定 |
|---|---|
| 平台 | Windows WSL Ubuntu |
| GPU | NVIDIA GeForce RTX 5070 |
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| SGLang version | sglang==0.5.9 |
| Torch | 2.9.1+cu128 |
| CUDA toolkit for JIT | user-space CUDA 12.8 nvcc |
| 用途 | 快速驗證 R1-R5、cache lookup logging、SubContextIndex prototype |

## 3. 測試模型

neno5 多模型實驗使用 Qwen2.5 Instruct 系列：

| 模型 |
|---|
| Qwen/Qwen2.5-0.5B-Instruct |
| Qwen/Qwen2.5-1.5B-Instruct |
| Qwen/Qwen2.5-3B-Instruct |
| Qwen/Qwen2.5-7B-Instruct |
| Qwen/Qwen2.5-14B-Instruct |

WSL 對照實驗使用：

| 模型 |
|---|
| Qwen/Qwen2.5-0.5B-Instruct |

## 4. Prompt 與 Sub-Context 設計

prompt 中的背景資料拆成三個 sub-context：

| Sub-context | 內容概念 |
|---|---|
| A | Product Alpha：適合 portable student developer 的 workstation laptop |
| B | Product Beta：適合 shared local inference lab 的 compact inference server |
| C | Product Gamma：適合 always-on automation node 的 low-cost mini PC |

每個 request 都使用相同 A/B/C 內容，但改變排列順序與最後的問題。

### 4.1 原始 R1/R2/R3 prefix-cache baseline

| Request | Sub-context order | 目的 |
|---|---|---|
| R1_ABC | A+B+C | 第一次送入完整上下文，作為 cold cache baseline |
| R2_ABC_same_prefix | A+B+C | 與 R1 使用相同 prefix，但問題不同，測試 prefix cache 是否命中 |
| R3_CAB_reordered | C+A+B | 相同 A/B/C 內容但重新排序，測試 reordered sub-context 是否可重用 |

預期：

1. R1 沒有 cache 命中。
2. R2 因與 R1 擁有相同 prefix，cache 命中率應很高。
3. R3 雖內容相同，但順序改變，因此原生 prefix cache 命中率會下降。

### 4.2 Step 5：Single-Context vs Sub-Context 對照實驗

後續對照實驗擴充成 R1-R5：

| Request | Sub-context order |
|---|---|
| R1_ABC | A+B+C |
| R2_ABC_same_prefix | A+B+C |
| R3_CAB_reordered | C+A+B |
| R4_BCA_reordered | B+C+A |
| R5_ACB_reordered | A+C+B |

比較三組方法：

| 實驗組 | 說明 |
|---|---|
| Baseline 1: No cache | 啟動 SGLang 時加入 `--disable-radix-cache` |
| Baseline 2: SGLang RadixAttention | 使用 SGLang 原生 RadixAttention prefix cache |
| Proposed: Sub-context-aware cache | 使用 metadata-level `SubContextIndex` prototype 估算 segment-level reuse |

觀察四個指標：

| 指標 | 定義 |
|---|---|
| cached token ratio | `cached_tokens / prompt_tokens` |
| prefill tokens | `prompt_tokens - cached_tokens` |
| first token latency | 非 streaming benchmark 中，以 `prefill_finished_ts - request_received_ts` 作為 TTFT proxy |
| total latency | SGLang `e2e_latency`，若缺少則使用 client-side latency |

## 5. Benchmark 與 Logging 實作

### 5.1 Benchmark script

`bench_sglang_prefix_cache.py` 會自動送出 R1-R5 request，並輸出：

```text
sglang_prefix_cache_<timestamp>.csv
sglang_prefix_cache_<timestamp>.json
sglang_prefix_cache_<timestamp>_subcontexts.csv
```

主 CSV/JSON 記錄：

- request name
- sub-context order
- prompt tokens
- cached tokens
- cached token ratio
- prefill tokens estimate
- first token latency
- total latency
- SGLang metrics delta
- cache lookup logging fields

### 5.2 Sub-context metadata

每個 request 會額外輸出 sub-context metadata：

| 欄位 | 意義 |
|---|---|
| request_id | R1、R2、R3、R4、R5 |
| subcontext_id | A、B、C |
| char_start / char_end | sub-context 在 prompt 字串中的位置 |
| token_start / token_end | tokenizer 後的 token span |
| token_len | sub-context token 數 |
| content_hash | sub-context 內容 hash |
| order | ABC、CAB、BCA、ACB |
| cached_tokens | SGLang 回傳 cached tokens |
| prompt_tokens | prompt token 數 |
| cache_ratio | cached_tokens / prompt_tokens |

### 5.3 精準 cache lookup logging

已在 SGLang runtime 的 prefix lookup path 加入 logging，記錄：

| 欄位 | 意義 |
|---|---|
| request_id / rid | SGLang request id |
| input_token_len | lookup 時輸入 token 數 |
| matched_prefix_len | RadixAttention 命中的 prefix token 數 |
| matched_node_id | 命中的 radix node id |
| cached_tokens | 命中 token 數 |
| uncached_tokens | 未命中 token 數 |
| first_mismatch_token_position | 第一個 mismatch 位置 |

WSL 小模型實測中，典型結果如下：

```text
R1_ABC:
  matched_prefix_len = 0

R2_ABC:
  matched_prefix_len = 346
  cache_ratio ~= 96%

R3_CAB:
  matched_prefix_len = 69
  first_mismatch_token_position = 69
```

## 6. neno5 多模型 Prefix Cache 實驗結果

### 6.1 Cached token ratio

| Model | R1_ABC | R2_ABC_same_prefix | R3_CAB_reordered |
|---|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 0.0% | 96.4% | 19.3% |
| Qwen2.5-1.5B-Instruct | 0.0% | 96.4% | 19.3% |
| Qwen2.5-3B-Instruct | 0.0% | 96.3% | 19.1% |
| Qwen2.5-7B-Instruct | 0.0% | 96.4% | 19.2% |
| Qwen2.5-14B-Instruct | 0.0% | 96.4% | 19.3% |

### 6.2 Cached token 數量

| Model | R1 cached tokens | R2 cached tokens | R3 cached tokens |
|---|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 0.0 | 351.3 | 70.3 |
| Qwen2.5-1.5B-Instruct | 0.0 | 350.0 | 70.0 |
| Qwen2.5-3B-Instruct | 0.0 | 340.0 | 67.5 |
| Qwen2.5-7B-Instruct | 0.0 | 346.0 | 69.0 |
| Qwen2.5-14B-Instruct | 0.0 | 350.0 | 70.0 |

### 6.3 Latency

latency 受生成 token 數、server warmup、GPU 狀態與排程影響，因此僅作輔助觀察，主要結論仍以 cached token ratio 與 cached token 數為主。

| Model | R1 latency (s) | R2 latency (s) | R3 latency (s) |
|---|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 1.417 | 0.806 | 0.462 |
| Qwen2.5-1.5B-Instruct | 0.616 | 0.961 | 0.377 |
| Qwen2.5-3B-Instruct | 0.432 | 0.686 | 0.598 |
| Qwen2.5-7B-Instruct | 0.453 | 0.771 | 0.529 |
| Qwen2.5-14B-Instruct | 0.809 | 0.751 | 0.618 |

## 7. Sub-Context-Aware Cache Prototype

本 prototype 不直接修改 SGLang Radix tree，而是在 RadixAttention prefix cache 外加一層 metadata-driven index：

```text
SubContextIndex:
  hash(A) -> token_range, kv_block_refs
  hash(B) -> token_range, kv_block_refs
  hash(C) -> token_range, kv_block_refs
```

當新 request 進來，例如：

```text
R3 = C + A + B
```

prototype 流程如下：

1. tokenize prompt
2. 根據 metadata 找出 C、A、B 的 token span
3. 對每個 span 讀取或計算 content hash
4. 查詢 `SubContextIndex`
5. hash 命中則標記該 span 為 reusable candidate
6. 沒命中的 span 走正常 prefill

這樣研究目標從「完整 prefix reuse」轉成「片段級 segment reuse」。

重要限制：

目前 prototype 只證明 metadata 可以辨識可重用 sub-context。它尚未證明任意 KV block 可以安全拼接或移動。若要成為真正 runtime feature，還需要處理：

- absolute / RoPE position 差異
- causal attention dependencies
- KV block ownership 與 lifetime
- prompt template 或 separator 改變時的 correctness check

因此安全說法是：

```text
SubContextIndex 可以在 RadixAttention prefix cache 外辨識 segment-level reuse candidates；
真正的 non-prefix KV reuse 仍需要 SGLang runtime 支援 position 與 attention dependency handling。
```

## 8. Step 5 對照實驗結果

本節使用 WSL + Qwen/Qwen2.5-0.5B-Instruct 小模型驗證三組方法：

```text
Baseline_NoCache
Baseline_RadixAttention
Proposed_SubContextIndex
```

### 8.1 Cached token ratio

| Request | No cache | RadixAttention | Proposed SubContextIndex |
|---|---:|---:|---:|
| R1_ABC | 0.0% | 0.0% | 0.0% |
| R2_ABC_same_prefix | 0.0% | 96.46% | 75.75% |
| R3_CAB_reordered | 0.0% | 19.35% | 75.75% |
| R4_BCA_reordered | 0.0% | 19.35% | 75.75% |
| R5_ACB_reordered | 0.0% | 44.57% | 75.54% |

### 8.2 Prefill tokens

| Request | No cache | RadixAttention | Proposed SubContextIndex |
|---|---:|---:|---:|
| R1_ABC | 354 | 366 | 366 |
| R2_ABC_same_prefix | 355 | 13 | 89 |
| R3_CAB_reordered | 355 | 296 | 89 |
| R4_BCA_reordered | 355 | 296 | 89 |
| R5_ACB_reordered | 356 | 204 | 90 |

### 8.3 First token latency

Proposed 目前是 metadata prototype，沒有實際執行 KV block reuse，因此 first token latency 欄位不宣稱實測改善。

| Request | No cache TTFT proxy (s) | RadixAttention TTFT proxy (s) | Proposed |
|---|---:|---:|---|
| R1_ABC | 2.6515 | 0.0208 | not measured |
| R2_ABC_same_prefix | 0.0212 | 0.0377 | not measured |
| R3_CAB_reordered | 0.0189 | 0.0191 | not measured |
| R4_BCA_reordered | 0.0186 | 0.0194 | not measured |
| R5_ACB_reordered | 0.0192 | 0.0194 | not measured |

### 8.4 Total latency

| Request | No cache total latency (s) | RadixAttention total latency (s) | Proposed |
|---|---:|---:|---|
| R1_ABC | 3.6039 | 0.3270 | not measured |
| R2_ABC_same_prefix | 0.5141 | 0.5393 | not measured |
| R3_CAB_reordered | 0.4783 | 0.4715 | not measured |
| R4_BCA_reordered | 0.3015 | 0.4278 | not measured |
| R5_ACB_reordered | 0.3742 | 0.3607 | not measured |

## 9. 結果分析

### 9.1 No cache baseline

No cache 組中，R1-R5 的 cached token ratio 全部為 0%。這驗證 `--disable-radix-cache` 正常生效，也提供了沒有 prefix cache 時的 prefill token baseline。

### 9.2 SGLang RadixAttention 對相同 prefix 非常有效

RadixAttention 組中，R2_ABC_same_prefix 的 cached token ratio 達到 96.46%，prefill tokens 只剩 13。這再次證明原生 prefix cache 對連續、同順序 prefix reuse 非常有效。

### 9.3 Reordered context 會降低原生 prefix cache reuse

R3_CAB 與 R4_BCA 中，RadixAttention cached token ratio 下降至 19.35%，prefill tokens 回升到 296。這表示原生 RadixAttention 主要依賴從 prompt 開頭開始的連續 prefix match；當 A/B/C 順序改變後，已經見過的 sub-context 無法被完整視為 reusable segments。

R5_ACB 的 RadixAttention ratio 較高，達 44.57%，是因為它仍以 A 開頭，能命中更多與 R1/R2 共同的開頭片段，但仍低於 R2 的 96.46%。

### 9.4 SubContextIndex 顯示 segment-level reuse opportunity

Proposed SubContextIndex 在 R2-R5 都能辨識 A/B/C 已出現過，因此 estimated cached token ratio 約 75.5% 至 75.8%，prefill tokens 約 89 至 90。

這個結果代表：

1. R3/R4/R5 雖然順序改變，但 sub-context hash 仍可命中。
2. Segment-level reuse 可補足原生 prefix cache 對順序敏感的限制。
3. 對 R3/R4，Proposed 的 reusable token estimate 明顯高於 RadixAttention。

但也要注意：

1. Proposed ratio 只計算 A/B/C sub-context 片段，不包含 prompt intro、question、template tokens。
2. Proposed latency 尚未被實測，因為目前還沒有真正 runtime KV block reuse。
3. 下一步要進入 runtime 層，才可以量測真正 first token latency 與 total latency 改善。

## 10. 結論

本實驗已完成 SGLang RadixAttention prefix cache baseline、sub-context metadata export、cache lookup logging、SubContextIndex prototype，以及 single-context vs sub-context 的三組對照實驗。

主要結論如下：

1. SGLang RadixAttention prefix cache 已成功啟用，且 metrics/log 可以觀察 cached tokens。
2. 相同 prefix 順序下，SGLang 原生 cache reuse 效果非常明顯，R2 可達約 96% cached token ratio。
3. 相同 sub-context 重新排序後，原生 prefix cache reuse 顯著下降，R3/R4 約 19%。
4. 此限制在 Qwen2.5 0.5B 到 14B 多個模型上皆一致存在。
5. SubContextIndex prototype 可以根據 content hash 辨識 reordered A/B/C，顯示 segment-level reuse 的潛在效益。
6. Proposed 方法目前只能安全宣稱 token-level reuse opportunity，不能宣稱已經實測 latency improvement。

## 11. 後續工作

下一階段建議分成三步：

1. 在 SGLang runtime 內建立真正的 sub-context index API。
2. 研究 non-prefix KV block reuse 的 correctness 條件，特別是 RoPE position、causal attention dependency、KV block lifetime。
3. 實作 minimal runtime prototype，讓 R3/R4/R5 中 hash 命中的 sub-context 能真的跳過部分 prefill，並重新量測 first token latency 與 total latency。

建議報告用語維持保守：

```text
目前 prototype 已證明 reordered sub-context 可被 metadata index 辨識為 reuse candidate；
實際 latency 加速仍需 SGLang runtime 支援 non-prefix KV cache reuse 後再量測。
```

## 12. 實驗資料來源

neno5 多模型資料來源：

```text
benchmark_results_qwen_20260507_162727/benchmark_results/
```

主要結果資料夾：

```text
neno5_Qwen__Qwen2.5-0.5B-Instruct_189905/
neno5_Qwen__Qwen2.5-0.5B-Instruct_189907/
neno5_Qwen__Qwen2.5-0.5B-Instruct_190019/
neno5_Qwen__Qwen2.5-1.5B-Instruct_189908/
neno5_Qwen__Qwen2.5-3B-Instruct_189909/
neno5_Qwen__Qwen2.5-3B-Instruct_190021/
neno5_Qwen__Qwen2.5-7B-Instruct_190598/
neno5_Qwen__Qwen2.5-14B-Instruct_190599/
```

WSL Step 5 對照實驗資料來源：

```text
benchmark_results/wsl_cache_baseline_matrix_manual_20260507_2213/
```

主要輸出檔：

```text
single_vs_subcontext_cache_comparison_20260507_221400.csv
single_vs_subcontext_cache_comparison_20260507_221400.json
subcontext_cache_prototype_20260507_221325.csv
subcontext_cache_prototype_20260507_221325.json
```

相關程式與文件：

```text
bench_sglang_prefix_cache.py
scripts/patch_sglang_cache_lookup_logging.py
scripts/subcontext_cache_prototype.py
scripts/compare_cache_baselines.py
scripts/wsl_run_cache_baseline_matrix.sh
docs/subcontext_cache_prototype.md
docs/single_vs_subcontext_experiment.md
```
