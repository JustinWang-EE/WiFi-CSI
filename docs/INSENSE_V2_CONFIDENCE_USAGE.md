# InSense V2 推論與信心輸出

在儲存庫根目錄執行 `python -m pip install -e .`，再執行：

```text
python -m wifi_densepose.confidence --csi window.npy --identity identity.json
```

- `window.npy` 含同一錄影連續 64 幀 CSIamp，形狀為 `[64,3,114,10]`。預設按訓練規格修補局部 NaN/Inf、逐幀 min–max 正規化。若輸入已經過 Stage7 正規化，使用 `--stage7` 並提供有限的 `[0,1]` 振幅。
- `identity.json` 提供 `subject`、`sequence`、`window` 三個非空字串及 `frames` 陣列。陣列須含順序連續的 64 個非負整數，例如 0 至 63，並與 NPY 的 frame 軸逐一對應。兩個 head 使用同一視窗的 D0 pose。軟體可驗證形狀與 frame IDs，擷取端負責正確綁定實際 CSI 與 IDs。
- CLI 預設使用 `configs/insense_v2_confidence.json` 與 `models/V2-root.pt`、`models/V2-posture.pt`、`models/V2-motion.pt`。可用 `--config`、`--root`、`--posture`、`--motion` 指定路徑。模型 SHA256 和類別順序仍受驗證。

輸出為單筆 JSON，可重導向到檔案。有效視窗的 `posture` 和 `motion` 分別提供 `predicted_class`、`confidence_pct` 及 `class_probabilities_pct`（四類／三類）。分數由 softmax(logits / T) 計算，Posture T=3.0716，Motion T=1.3749，屬探索性縮放。

`candidate_state` 和 `state_name` 由兩個 Top-1 類別按[十態映射表](INSENSE_V2_MODEL_ARCHITECTURE.md)決定，B3 的三個 Motion 分支共用 S30。發布介面只提供候選類別與名稱。發布設定為 `UNVERIFIED`，有效輸入的 `status` 為 `UNCERTAIN`，`state` 為 null，兩個 head 的百分比仍完整呈現。UNCERTAIN 代表系統未接受此次判斷。缺幀、錯序或無效 CSI 會回傳 `INVALID_INPUT` 和空候選。

[信心度評估](INSENSE_V2_CONFIDENCE_EVALUATION.md)中的門檻屬離線探索，未啟用於發布設定。
