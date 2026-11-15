# InSense V2 信心介面使用方式

使用具備專案既有依賴的 Python，從儲存庫根目錄執行。模型及 CSI 路徑由使用者提供；完整 SHA256 見[模型架構](INSENSE_V2_MODEL_ARCHITECTURE.md)。

```bat
set PYTHONPATH=src
python -m wifi_densepose.confidence --config configs/insense_v2_confidence.json --root models/V2-root-testing.pt --posture PATH_TO_D0_K4 --motion PATH_TO_D0_M3 --csi PATH_TO_WINDOW_NPY --identity PATH_TO_IDENTITY_JSON
```

CSI 陣列為 `[64,3,114,10]`，從同一段錄影擷取連續 64 幀，已用 Stage7 處理的輸入請加 `--stage7`。身分 JSON 包含 `subject`、`sequence`、`window` 三個非空字串和 `frames` 陣列。`frames` 須依序列出同一錄影中的 64 個連續非負整數，例如 Python 的 `list(range(32, 96))`。輸出保留完整識別、每個 head 的 logits、依固定類別排序的機率、Top-1 類別、信心度、Top-1 和 Top-2 機率之差、模型與 taxonomy SHA256，以及版本和狀態。

| 狀態 | 意義 |
|---|---|
| `VALID` | 已驗證的設定中，必要的 head 皆通過信心與 margin 門檻，回傳對應狀態代碼。門檻相等時接受。 |
| `UNKNOWN` | 校準未驗證，或其必要 head 信心不足。回傳視窗身分與兩個 head 的原始結果，`state` 為空。 |
| `INVALID_INPUT` | 資料缺幀、排列錯誤、形狀或數值無效，或兩個 head 的視窗或類別不是同一合約。 |

發布設定為 `UNVERIFIED`、溫度為 1、門檻為空。有效輸入仍會輸出未校準的兩個 head 分數，但組合狀態為 `UNKNOWN`。`UNKNOWN` 不表示安全；應保留這筆記錄供後續處理。驗證實際部署所需的獨立校準資料後，方可制定另外的版本設定。跳躍覆蓋規則和完整十狀態表見[模型架構](INSENSE_V2_MODEL_ARCHITECTURE.md)。

## 重現 CPU 稽核與探索校準

下載和準備資料遵循既有資料合約。只使用下列程式指定的共同專案 DEV，不得用官方 TEST 來選溫度或門檻。`CACHE_DIR` 指既有 D0 下游快取、`L3_DIR` 指既有 Motion L3 split 和標籤、`STAGE7_DIR` 指 Stage7 快取、`K4_SEQUENCES_CSV` 指來源 Action27 sequence metadata。

```bat
python scripts\export_insense_confidence_dev.py --config configs\insense_v2_confidence.json --root models\V2-root-testing.pt --posture PATH_TO_D0_K4 --motion PATH_TO_D0_M3 --cache CACHE_DIR --l3 L3_DIR --stage7 STAGE7_DIR --posture-sequences K4_SEQUENCES_CSV --output outputs\confidence_audit
scripts\evaluate_insense_confidence.bat outputs\confidence_audit\common_dev_confidence.jsonl outputs\confidence_analysis --fit-subjects S04 S24 --eval-subjects S14 S34
python -m unittest discover -s tests -p test_insense_confidence.py
```

評估 BAT 可用 `INSENSE_PYTHON` 指定既有 Python。它只把實驗校準摘要與兩張 reliability SVG 寫入新目錄；不更新發布設定。輸出包含每個覆蓋目標的擬合門檻、實際覆蓋率、接受與拒絕數、接受後指標、含拒絕的類別召回率及逐受試者結果。應保留 JSONL、JSON 和 SVG 在本機，不提交預測、資料集或快取。數值與限制見[信心稽核](INSENSE_V2_CONFIDENCE_REPORT.md)。
