# SGLang RadixAttention Prefix Cache 實驗報告

## 摘要

本實驗目標是確認 SGLang 的 RadixAttention prefix cache 是否能有效重用重複的上下文，並觀察當相同 sub-context 被重新排序時，cache 命中率是否會下降。實驗結果顯示，當 request 使用相同順序的 prefix 時，SGLang 可以達到約 96% 的 cached token ratio；但當相同內容被重新排序後，cached token ratio 下降至約 19%。這表示目前的 RadixAttention prefix cache 對 prefix 順序高度敏感，尚無法直接重用位置改變後的 sub-context。

此結果支持後續研究方向：在 SGLang 中設計 sub-context-aware caching，使系統能辨識並重用 A、B、C 等獨立上下文片段，而不只依賴完整 prefix 的連續匹配。

## 1. 實驗目的

本實驗根據教授給定的方向，先完成 OpenClaw 與 SGLang runtime 的初始環境建置，並啟用 SGLang 的 RadixAttention prefix cache 與 request/KV cache 相關 logging。接著設計 R1、R2、R3 三組 request，自動送入 SGLang server，收集 metrics 與 log，並輸出 CSV/JSON 作為後續分析資料。

主要驗證問題如下：

1. SGLang RadixAttention prefix cache 在相同 prefix 下是否能有效命中？
2. 當相同 sub-context 重新排序時，prefix cache 是否仍能重用？
3. 不同模型大小是否會改變 prefix cache 對順序的敏感性？

## 2. 實驗環境

實驗於 neno5 GPU 節點上執行，並使用 SLURM job 啟動 SGLang server 與 benchmark script。

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
| Logging | SGLang request logging / metrics enabled |

SGLang server 啟動時使用的主要設定包含：

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

## 3. 測試模型

本次多模型實驗使用 Qwen2.5 Instruct 系列模型，從小模型到較大模型皆包含在測試中：

| 模型 |
|---|
| Qwen/Qwen2.5-0.5B-Instruct |
| Qwen/Qwen2.5-1.5B-Instruct |
| Qwen/Qwen2.5-3B-Instruct |
| Qwen/Qwen2.5-7B-Instruct |
| Qwen/Qwen2.5-14B-Instruct |

## 4. 實驗設計

本實驗將 prompt 中的背景資料拆成三個 sub-context，分別記為 A、B、C。三個 sub-context 描述三種不同產品或節點情境，並在不同 request 中改變問題與排列順序。

三組 request 設計如下：

| Request | Sub-context order | 目的 |
|---|---|---|
| R1_ABC | A+B+C | 第一次送入完整上下文，作為 cold cache baseline |
| R2_ABC_same_prefix | A+B+C | 與 R1 使用相同 prefix，但問題不同，用來測試 prefix cache 是否命中 |
| R3_CAB_reordered | C+A+B | 使用相同 A/B/C 內容但重新排序，用來測試 cache 是否能重用 reordered sub-context |

如果 SGLang 只能匹配連續 prefix，則預期結果會是：

1. R1 沒有 cache 命中。
2. R2 因為與 R1 擁有相同 prefix，因此 cache 命中率應該很高。
3. R3 雖然內容相同，但因順序不同，cache 命中率會顯著降低。

## 5. 實驗結果

### 5.1 Cache ratio 比較

下表整理不同模型在 R1、R2、R3 下的平均 cached token ratio。

| Model | R1_ABC | R2_ABC_same_prefix | R3_CAB_reordered |
|---|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 0.0% | 96.4% | 19.3% |
| Qwen2.5-1.5B-Instruct | 0.0% | 96.4% | 19.3% |
| Qwen2.5-3B-Instruct | 0.0% | 96.3% | 19.1% |
| Qwen2.5-7B-Instruct | 0.0% | 96.4% | 19.2% |
| Qwen2.5-14B-Instruct | 0.0% | 96.4% | 19.3% |

### 5.2 Cached token 數量比較

下表整理不同模型在各 request 下的平均 cached token 數量。

| Model | R1 cached tokens | R2 cached tokens | R3 cached tokens |
|---|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 0.0 | 351.3 | 70.3 |
| Qwen2.5-1.5B-Instruct | 0.0 | 350.0 | 70.0 |
| Qwen2.5-3B-Instruct | 0.0 | 340.0 | 67.5 |
| Qwen2.5-7B-Instruct | 0.0 | 346.0 | 69.0 |
| Qwen2.5-14B-Instruct | 0.0 | 350.0 | 70.0 |

### 5.3 Latency 比較

下表整理不同模型在各 request 下的平均 latency。由於每次生成 token 數與 cluster 當下狀態可能略有差異，latency 僅作為輔助觀察，主要結論仍以 cached token ratio 與 cached token 數量為主。

| Model | R1 latency (s) | R2 latency (s) | R3 latency (s) |
|---|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 1.417 | 0.806 | 0.462 |
| Qwen2.5-1.5B-Instruct | 0.616 | 0.961 | 0.377 |
| Qwen2.5-3B-Instruct | 0.432 | 0.686 | 0.598 |
| Qwen2.5-7B-Instruct | 0.453 | 0.771 | 0.529 |
| Qwen2.5-14B-Instruct | 0.809 | 0.751 | 0.618 |

## 6. 結果分析

### 6.1 R1 顯示 cold cache baseline

所有模型在 R1_ABC 的 cached token ratio 都是 0.0%。這符合預期，因為 R1 是第一次送入 A+B+C 上下文，server 尚未累積可重用的 prefix cache。

### 6.2 R2 顯示 RadixAttention 對相同 prefix 有高命中率

R2_ABC_same_prefix 與 R1 使用相同的 A+B+C prefix，只改變最後的問題。結果顯示，不同模型的 R2 cached token ratio 都約為 96.3% 至 96.4%。這代表 SGLang 的 RadixAttention prefix cache 能有效重用前一個 request 的 prefix tokens。

這個結果證明目前環境中的 RadixAttention prefix cache 已成功啟用，而且 request logging 與 metrics 能正確反映 cached tokens。

### 6.3 R3 顯示 reordered sub-context 無法被完整重用

R3_CAB_reordered 使用與 R1/R2 相同的 A、B、C 內容，但順序改成 C+A+B。結果顯示，R3 的 cached token ratio 只剩約 19.1% 至 19.3%。這表示 SGLang 仍然只命中了 prompt 開頭共同模板或部分固定文字，而沒有完整重用已經出現過的 A、B、C sub-context。

此現象說明目前的 prefix cache 是 order-sensitive 的。只要上下文順序改變，即使內容相同，RadixAttention 也無法把重新排列後的 sub-context 當成已快取片段來重用。

### 6.4 模型大小不改變此現象

0.5B、1.5B、3B、7B、14B 模型都呈現相同模式：

1. R1 約 0% cache hit。
2. R2 約 96% cache hit。
3. R3 約 19% cache hit。

因此，cache 命中差異主要不是來自模型大小，而是來自 prefix cache 機制本身對 token sequence order 的限制。這使得實驗結論更穩定，也更能支持後續 sub-context cache 設計。

## 7. 結論

本實驗成功完成 OpenClaw 與 SGLang runtime 的初始建置，並在 neno5 H100 GPU 節點上啟用 SGLang RadixAttention prefix cache 與 request/KV cache logging。實驗結果證明，SGLang 對相同順序的 prefix 具有非常高的 cache reuse 能力，R2 的 cached token ratio 約為 96%。然而，當相同 sub-context 重新排序後，R3 的 cached token ratio 下降至約 19%。

因此，本實驗可以得到以下結論：

1. SGLang RadixAttention prefix cache 已成功啟用並能被 metrics/log 觀察。
2. 相同 prefix 順序下，cache reuse 效果非常明顯。
3. 相同 sub-context 重新排序後，cache reuse 顯著下降。
4. 此限制在不同模型大小上皆一致存在。
5. 後續若要支援更彈性的 context reuse，需要設計 sub-context-aware caching，而不是只依賴完整 prefix 的 Radix tree 匹配。

## 8. 後續工作

下一階段可以從 benchmark script 與 SGLang runtime instrumentation 兩個方向進行。

首先，在 benchmark script 中加入 sub-context span metadata，記錄 A、B、C 在 prompt 中的字元範圍、token 範圍、hash 與排列順序。這樣可以明確知道每個 request 中哪些 token 屬於同一個 sub-context。

接著，在 SGLang runtime 中加入更細緻的 cache lookup logging，記錄每個 request 在 RadixCache lookup 時命中的 prefix 長度、matched node、cached token 數量與未命中的 token 範圍。這能幫助確認 R3 為何只命中約 19%，並找出哪些 token 被 cache 重用。

最後，可以設計 sub-context-aware cache prototype。其核心想法是將 A、B、C 這些可獨立重用的上下文片段建立 fingerprint 或 hash index，讓系統在 reordered context 中也能找到已出現過的 sub-context，進一步提升非連續或重新排序情境下的 cache reuse。

## 9. 實驗資料來源

本報告使用以下 neno5 benchmark 結果資料：

```text
benchmark_results_qwen_20260507_162727/benchmark_results/
```

主要結果檔案包含：

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
