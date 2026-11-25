# InSense V3 模型架構與介面

**InSense V3 發布版本；exploratory / non-independent DEV、seed 0。發布過程沒有重新訓練、量化或 STM32 部署。**

三個 V3 檔案是指定最佳 checkpoint 的逐位元組副本。原始實驗及 V2 檔案保留；檔內歷史名稱不改寫，V3 名稱由檔名與設定提供。總參數 **133,842**，較正式 D0 V2 的 322,842 減少 **58.54%**。

| V3 model | Source selection | Best epoch | Parameters | Checkpoint SHA256 |
|---|---|---|---|---|
| root | Small-B CNN | 2 | 49,371 | `c42c0047f95ae7a631581b9b95bdb6aedcd9fb08a36ab63f544679d448ee3e75` |
| posture | K4 Small-C (48ch/3) | 25 | 51,156 | `6971a90e05403b62e8324faa82af1ab0c05bb10b598b1124633f9a587b20b83c` |
| motion | M3 Small-B (32ch/4) | 26 | 33,315 | `1188bfffe910395179a85f4abc161c5e937ff297bc76cbc25eb034623b4a881b` |

## 架構與張量

- **Root V3**：`[N,3,114,10] → [N,17,3]`。Conv2D 3→12→24→32，全部 kernel3、stride1、padding1，每層後 ReLU；AdaptiveAvgPool2D(4,2)；Flatten 256；Linear 256→128、ReLU、Linear 128→51；重排 17×3。
- **Posture V3**：`[N,48,64] → [N,4]`。Conv1D 1×1 投影 48→48；3 residual blocks，dilation1/2/4。每個 block 有兩層 kernel3 Conv1D（stride1、padding=dilation），各接 GroupNorm(8)、GELU、dropout0.20，再加 residual。mean/max 串接 96；Linear96→64、GELU、dropout0.20、Linear64→4。
- **Motion V3**：`[N,48,64] → [N,3]`。Conv1D 1×1 投影 48→32；4 residual blocks，dilation1/2/4/8，同樣的 GN8/GELU/dropout/residual。mean/max 與相鄰隱藏特徵絕對差的均值串接 96；Linear96→64、GELU、dropout0.20、Linear64→3。
- 推論一律 `eval()`、凍結參數，dropout 關閉。Head 的 symmetric padding 與 GroupNorm 使用完整視窗；目前介面需要緩衝 64 幀，不宣稱逐幀 causal TCN。

## 64 幀 CSI 串接

```text
連續 64 幀 CSI [64,3,114,10]
  → 共用 Root V3 一次，得到 [64,17,3]
  → 移除 joint0，保留 joints1–16，按 joint-major XYZ 排成 [1,48,64]
  ├→ Posture 的 Small-B TRAIN Mean/Std → K4 logits [1,4]
  └→ Motion 的 Small-B TRAIN Mean/Std  → M3 logits [1,3]
  → 各自 softmax 百分比 → 兩個 Top1 的十態候選映射
```

CSI 處理直接重用 V2 的 `preprocess_csi`：raw amplitude 每個尾軸 slice 以有限值均值補 NaN/Inf，接單幀全域 min–max。缺少整個 slice 或退化 raw frame 失敗。Stage7 模式只接受有限、實數且範圍 [0,1] 的資料，不再次 min–max。`WindowID` 要求同一段 64 個連續、依序、非負 frame ID；缺幀、錯序、錯 shape 回傳 INVALID_INPUT。

Root 輸出使用原生 XYZ、metres，已在 pelvis-relative 訓練座標中。GT 評估才計算 Q=P−P0；預測值不再減 predicted pelvis，也不強制將預測 joint0 清零。Head 僅丟棄 joint0。Root17 MPJPE 計入全部 17 關節，不做 PA alignment。

關節順序：0:pelvis, 1:right_hip, 2:right_knee, 3:right_ankle, 4:left_hip, 5:left_knee, 6:left_ankle, 7:spine, 8:thorax, 9:neck, 10:head, 11:left_shoulder, 12:left_elbow, 13:left_wrist, 14:right_shoulder, 15:right_elbow, 16:right_wrist。中央鏈名稱依已稽核 topology 為 spine/thorax/neck/head，保留原資料數值軸，沒有新增物理軸轉換。

## 正規化與固定類別

設定檔為 [configs/insense_v3.json](../configs/insense_v3.json)。Checkpoint 路徑相對 package 根目錄；含三個 SHA256、架構、完整兩套 48 維 Mean/Std、fit cohort、類別順序、joint order 及十態表。載入器檢查 SHA256、epoch、參數量、架構、Root/taxonomy 綁定，並驗證設定的統計量逐值等於 head checkpoint。V2 的統計量或模型檔不能通過 V3 allowlist。

- Posture：24 TRAIN subjects、648 recordings、192,456 個原錄影幀，每幀一次；std floor1e-6。
- Motion：28 TRAIN subjects、6804 個原 TRAIN windows、435,456 window-frame positions；沿用原規則包含被分類排除的 TRAIN windows，std floor1e-6。
- Posture 順序：`B0_UPRIGHT_ACTIVITY, B1_LOWER_BODY_ACTIVITY, B2_BENT_OR_LOW, B3_JUMP`。
- Motion 順序：`S0_STATIC, S1_IN_PLACE_ACTIVITY, S3_POSTURE_TRANSITION`；索引2為 Transition。

七個百分比為 **T=1 原始 softmax**，Posture 四類總和100%，Motion 三類總和100%。沒有套用 V2 探索性溫度，沒有設定 confidence 門檻。Softmax 百分比不是已校準的安全事件機率。

| Posture | Static index0 | Activity index1 | Transition index2 |
|---|---|---|---|
| B0 | S00 | S01 | S02 |
| B1 | S10 | S11 | S12 |
| B2 | S20 | S21 | S22 |
| B3 | S30 | S30 | S30 |

原規則使用兩個 head 各自 Top1 再查表。B3 的三個 Motion 分支一律 S30。沒有改用十格乘積的最大值，也不新增分類器。`status=VALID` 代表輸入／模型輸出有效；`candidate_only=true` 明示候選狀態，並不代表真實安全事件經過驗證。映射測試通過不等於已量得整合十分類 Accuracy。

## 與正式 V2 的 DEV 配對比較

V2 使用正式 D0 + K4 Epoch15 + M3 Epoch4。V2 數值重用既有、同一組 ID/標籤/視窗的配對重播，不宣稱本次重新跑過 V2 CSI。V3 本次實際重播 216 recordings ×297=64,152 個 CSI 幀，Root GPU、heads CPU；建構1944個 Posture window contexts，另重播12個真實 CSI CPU API windows（四位共同 DEV subjects 的 A01/A12/A26）。

Root17：V2 **112.632184 mm**；V3 **112.063222 mm**，差值 **-0.568962 mm**。

| 評估單位 | V2 Acc | V3 Acc | Δ Acc pp | V2 Macro-F1 | V3 Macro-F1 | Δ F1 pp |
|---|---|---|---|---|---|---|
| Posture: 216 sequences / 8 subjects | 69.91% | 68.06% | -1.85 | 52.94% | 57.54% | +4.61 |
| Posture: 972 windows / 4 subjects | 63.79% | 57.61% | -6.17 | 43.42% | 42.79% | -0.63 |
| Motion: 850 eligible windows / 4 subjects | 57.88% | 61.18% | +3.29 | 58.17% | 61.36% | +3.20 |

216 sequences：S04,S09,S14,S19,S24,S29,S34,S39；每錄影九窗 logits 平均後 argmax。972 Posture windows 與850 Motion eligible windows：S04,S14,S24,S34。各窗長64，start0/32/64/96/128/160/192/224/233。850為既有 GT geometry eligibility 篩選後的子集，沒有重新產生標籤。兩種 Posture 單位與受試者集合不同，不能互換。

## 每類 Recall 與分類風險

### Posture 216 sequences

| Class | Support | V2 Recall | V3 Recall |
|---|---|---|---|
| B0_UPRIGHT_ACTIVITY | 120 | 84.17% | 82.50% |
| B1_LOWER_BODY_ACTIVITY | 56 | 73.21% | 64.29% |
| B2_BENT_OR_LOW | 32 | 21.88% | 25.00% |
| B3_JUMP | 8 | 25.00% | 50.00% |

V3 Confusion Matrix：GT rows、predicted columns，順序同表。

```text
99 12 8 1
13 36 6 1
19 5 8 0
3 1 0 4
```

### Posture 972 windows

| Class | Support | V2 Recall | V3 Recall |
|---|---|---|---|
| B0_UPRIGHT_ACTIVITY | 540 | 79.07% | 77.78% |
| B1_LOWER_BODY_ACTIVITY | 252 | 66.67% | 44.44% |
| B2_BENT_OR_LOW | 144 | 13.19% | 10.42% |
| B3_JUMP | 36 | 16.67% | 36.11% |

V3 Confusion Matrix：GT rows、predicted columns，順序同表。

```text
420 53 66 1
78 112 43 19
104 25 15 0
13 10 0 13
```

### Motion 850 windows

| Class | Support | V2 Recall | V3 Recall |
|---|---|---|---|
| S0_STATIC | 264 | 65.53% | 77.27% |
| S1_IN_PLACE_ACTIVITY | 358 | 54.47% | 52.51% |
| S3_POSTURE_TRANSITION | 228 | 54.39% | 56.14% |

V3 Confusion Matrix：GT rows、predicted columns，順序同表。

```text
204 53 7
59 188 111
18 82 128
```

Posture Jump：216-sequence 為 **4/8=50.00% Recall**，972-window 為 **13/36=36.11%**。B2 Recall 分別只有25.00%及10.42%；不能只看總體 Accuracy。Motion Transition 為 **128/228=56.14% Recall**，仍漏判100個 eligible transition windows；Activity Recall52.51%。三個評估單位均有預測到所有類別，無單一類別塌縮。

與 V2 相比，V3 的 sequence Posture Macro-F1 與 Motion 指標提高，但 Posture sequence Accuracy 以及共同972-window表現降低，不能宣稱所有任務全面改善。

所有選模已反覆使用 DEV；single seed、非獨立評估。Motion TRAIN cohort 含 S09/S19/S29/S39，這四位亦在 Root/Posture DEV；共同四位 window DEV 不在該 Motion TRAIN cohort。各任務成績應分開解讀，不能當作同一獨立八人整合安全評估。未使用 official TEST，沒有實際居家安全資料驗證。

## 資源比較與 STM32 限制

| Role | V2 Params | V3 Params | 減少 | V2 MACs | V3 MACs | 減少 | MAC單位 |
|---|---|---|---|---|---|---|---|
| root | 95,827 | 49,371 | 48.48% | 26,830,144 | 11,243,216 | 58.09% | frame |
| posture | 111,492 | 51,156 | 54.12% | 6,496,512 | 2,808,064 | 56.78% | 64-frame window |
| motion | 115,523 | 33,315 | 71.16% | 6,500,544 | 1,677,504 | 74.19% | 64-frame window |

完整64幀區間（Root每幀一次＋兩個heads各一次）：V2 **1,730,126,272 MACs**；V3 **724,051,392 MACs**，減少 **58.15%**。不代表每新增一幀都必須重算64次 Root；連續系統可重用已算 Pose。

MACs為實際 forward hook 計數的 Conv/Linear 乘加；不計 GN/GELU/ReLU/pooling、bias addition及residual addition。V3 FP32 理論權重為535,368 bytes（522.82 KiB），V2為1,291,368 bytes（1261.10 KiB），不含activation、buffer、runtime及checkpoint optimizer/RNG metadata。

STM32 的峰值 RAM、Flash映像、latency、功耗、Conv1D/GN/GELU實作與數值一致性 **均未量測**。目前需要64幀buffer，實際端到端延遲亦未量測。沒有 INT8 量化或部署；理論參數與MAC減少不構成STM32可部署證明。

## 使用方式

在專案根目錄啟用既有、含 numpy/torch 的 Python 環境：

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m wifi_densepose.v3_inference `
  --config configs/insense_v3.json `
  --csi path/to/csi.npy --identity path/to/identity.json `
  --stage7 --output path/to/new_output.json
```

`csi.npy`為[64,3,114,10]；raw amplitudes請省略`--stage7`。identity JSON包含subject、sequence、window三個非空字串，以及64個連續frame IDs。輸出包含四個Posture百分比、三個Motion百分比、兩個Top1、候選state、模型版本綁定。輸出檔使用exclusive create，避免覆蓋。

```json
{"subject":"Sxx","sequence":"recording-id","window":"0","frames":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]}
```

frames須填入完整64個連續整數。所有model/config路徑均以package相對位置解析，不含個人硬編碼路徑。

## 驗證結果與證據

- 原始checkpoint與V3副本SHA256、全部model tensors完全一致；三模型strict load PASS。
- V3實作對原始實驗模型的CPU輸出差異均為0。
- 本次Root重播對cached Pose最大差異0m，兩任務正規化後特徵最大差異皆0。
- 參考logits最大差異3.81469727e-06，預先容許atol=0.002；1944 Posture與850 Motion預測一致。
- 12個CPU真實視窗百分比最大差異0.001826662pp，容許0.02pp；類別與state一致。CLI額外執行同一個真實視窗成功，未增加獨立視窗數。
- 12個映射分支、972個真實視窗映射及七個百分比總和檢查PASS；26項unit tests PASS，含V2回歸、V3相對路徑搬移及發布訓練紀錄檢查。
- 原始實驗及 V2 checkpoint/config/inference/documentation hash維持不變；README主版本更新為V3。

發布的真實訓練紀錄：[Root 10 epochs](../training_logs/v3/root.json)、[Posture 30 epochs](../training_logs/v3/posture.json)、[Motion 30 epochs](../training_logs/v3/motion.json)。各檔記錄完整原始CSV數值、CSV SHA256、選模規則、最佳Epoch及checkpoint身分；不重新生成訓練結果。原始MM-Fi資料、特徵快取、逐窗資料與本機稽核路徑不隨發布上傳。

模型程式：[models/insense_v3.py](../src/wifi_densepose/models/insense_v3.py)；入口：[v3_inference.py](../src/wifi_densepose/v3_inference.py)；測試：[test_v3_inference.py](../tests/test_v3_inference.py)。
