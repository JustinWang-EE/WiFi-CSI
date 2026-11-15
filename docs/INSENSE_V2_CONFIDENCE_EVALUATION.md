# InSense V2 Posture / Motion 信心度評估

**Exploratory / non-independent DEV。** 使用凍結 D0、K4、M3 的保存 logits，按受試者、錄影、視窗索引和 64 個連續 frame IDs 核對類別順序、taxonomy 和權重 SHA256。972 個視窗有 K4 Action27 代理真值，其中 850 個同時有凍結的 GT geometry Motion 標籤。K4 指標以視窗為單位。這些重疊視窗來自曾用於開發或選模的 DEV，兩個代理標籤均無法證實居家安全事件辨識能力。

## 無門檻結果

| 模型 | 全 DEV N | Accuracy | Macro-F1 | Balanced Accuracy | Evaluation N | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Posture | 972 | 63.79% | 43.42% | 43.90% | 486 | 64.61% | 38.72% |
| Motion | 850 | 57.88% | 58.17% | 58.13% | 419 | 55.85% | 55.46% |

Raw 與 temperature-scaled 使用同一批 logits，T=3.0716（Posture）與 1.3749（Motion）。兩者無門檻時 Top-1 完全相同。溫度源自 S04、S24，門檻也只用這兩位受試者選擇。S14、S34 用於 evaluation。Macro-F1 固定使用所有類別，零分母 F1 計 0。Balanced Accuracy 平均有真值支持的類別召回率。

## Fitting 選定門檻與 80% 目標

受測門檻為無門檻、50%、60%、70%、80%、90%、95%。各 head 在 fitting 選擇達到 80% Accuracy 且保留最多視窗的門檻，並列取較低門檻。雙 head 對 Raw 和 scaled 分別掃描 49 個組合，要求兩頭在 fitting 各自達 80% 並最大化共同保留數，並列時依序比較門檻和及 Posture 門檻。

| 模型／分數 | 門檻 | Evaluation 保留／總數 | Coverage | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Posture Raw | fitting 無可用門檻 | — | — | — | — |
| Posture scaled | 70% | 162/486 | 33.33% | 77.16% | 38.42% |
| Motion Raw | 70% | 108/419 | 25.78% | 76.85% | 34.63% |
| Motion scaled | 70% | 85/419 | 20.29% | 88.24% | 31.25% |

Posture 的 95% scaled 門檻在描述性掃描保留 10/486（2.06%）並達 80.00%，只剩 B0 和 B2 兩類真值。此門檻未由 fitting 選定。70% 的 Posture 門檻將 B1 從 126 筆濾到 2 筆、B3 從 18 筆濾到 2 筆，保留的 B2 和 B3 召回率皆為零。Motion 的 70% scaled 門檻保留 Static、Activity、Transition 真值 75、5、5 筆，召回分別為 100%、0%、0%。

| Joint score | Posture／Motion 門檻 | 共同保留／419 | Coverage | Posture Accuracy | Motion Accuracy | 兩頭皆正確 |
|---|---|---:|---:|---:|---:|---:|
| Raw（fitting） | 無／70% | 108 | 25.78% | 87.96% | 76.85% | 75.00% |
| Scaled（fitting） | 70%／60% | 87 | 20.76% | 89.66% | 79.31% | 79.31% |
| Scaled（固定 70% 參照） | 70%／70% | 68 | 16.23% | 92.65% | 89.71% | 89.71% |

Fitting 選定的 scaled 組合中，Posture 真值支持為 77／1／9／0，Motion 為 69／9／9。固定 70% 參照只保留 16.23%，Posture B1、B3 無真值支持，Motion Activity 和 Transition 的召回仍為零。兩頭皆正確以共同保留且有雙真值的視窗為分母。結果顯示高分可篩出以 B0 或 Static 為主的子集，現有資料不足以支持部署門檻。發布設定保持 UNCERTAIN。

## 推論驗證

八個真實 Stage7 視窗完成嚴格權重載入和 CPU CSI → Root → 兩頭推論。與保存參考值相比，最大 pose 差 2.385×10⁻⁷（容差 5×10⁻⁶），Posture／Motion 最大 logit 差分別為 1.762×10⁻⁴／1.078×10⁻⁴（容差 0.002），縮放機率最大差為 0.00210／0.00155 個百分點（容差 0.1）。類別順序、兩頭時間對齊及 Top-1 一致。逐窗評估與混淆矩陣等完整資料保存在本機實驗輸出。
