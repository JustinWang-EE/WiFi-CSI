# InSense V2 模型架構

信心介面版本為 `D0-K4-M3-confidence-v1`。對同一段連續 64 幀 CSI，只推論一次 D0，再以兩套訓練集正規化各自送入 K4 和 M3。根節點與兩個分類器合計 **322,842** 個參數。

| 角色 | Checkpoint 身分 | 參數 | SHA256 |
|---|---|---:|---|
| V2-root | D0 Epoch 1 | 95,827 | `267314703a410c8d6d6c7dd95911fb30fad39c0b8c83fe539538d494c5d8748b` |
| V2-posture | D0 K4 延伸實驗的 Epoch 15 | 111,492 | `59fbdae69be174430307bf73b2d1dd960fb0f5e651d1090b75af1b2716204667` |
| V2-motion | D0 M3 十 epoch 實驗的 Epoch 4 | 115,523 | `4d7b078bb690b406b695db88bc8153521d5fe3c92c6031488e2a343ea629007e` |

目前儲存庫的 `models/V2-root-testing.pt` 符合 D0 雜湊。K4、M3 需另行指定上表的原始 D0 相容 checkpoint。既有 `models/V2-posture.pt`、`models/V2-motion.pt` 屬於早期的 R03／ROOT9A 組合，載入器會拒絕其 SHA256。此版發布程式、設定與文件；模型來源和版本以 SHA256 驗證。

## 推論規格

- **D0**：`[64,3,114,10]` 連續 CSIamp。單幀先依尾軸切片，以有限值平均補 NaN/Inf，再做單幀全域 min–max。Stage7 已處理輸入可直接供應 `[0,1]` 的有限值。每幀經 3 層 `3×3` Conv2D（3→16→32→64，stride 1、padding 1）與 ReLU、`4×2` adaptive average pooling、Linear 512→128→51，重排為 `[64,17,3]`，約 26,830,144 Conv/Linear MACs／幀。
- **共同姿態介面**：沿用 D0 原生 root-relative XYZ 與關節順序，拿掉 joint 0，不再扣除預測 pelvis。關節 1–16 的 XYZ 按關節優先排成 `[48,64]`。K4 正規化由 24 位 Posture TRAIN 受試者、原錄影逐幀擬合；M3 由 28 位 Motion TRAIN 受試者的原視窗擬合，包含既有排除視窗。設定保留兩套統計量。
- **K4 和 M3**：`1×1` 投影 48→64，4 個雙 `3×1` 一維卷積殘差區塊（dilation 1、2、4、8）。每層含 GroupNorm(8)、GELU 與訓練時 dropout 0.20。K4 串接 mean/max pooling 後經 Linear 128→64→4。M3 另串接相鄰隱藏向量的絕對差均值，經 Linear 192→64→3。推論關閉 dropout。K4 在共同視窗介面輸出每窗 logits；原選模仍是九窗 logits 平均後評估。

## 固定類別順序與十狀態

姿態依序為 `B0_UPRIGHT_ACTIVITY`、`B1_LOWER_BODY_ACTIVITY`、`B2_BENT_OR_LOW`、`B3_JUMP`。動態依序為 `S0_STATIC`、`S1_IN_PLACE_ACTIVITY`、`S3_POSTURE_TRANSITION`。動態類別索引 2 是 Transition。

| 姿態 | Static | Activity | Transition |
|---|---|---|---|
| B0 | S00 | S01 | S02 |
| B1 | S10 | S11 | S12 |
| B2 | S20 | S21 | S22 |
| B3 | S30 | S30 | S30 |

達到姿態門檻的 B3 跳躍映射為 S30，與動態類別無關。兩個 head 的視窗身分與排序仍須一致。共有三個神經網路，十狀態由確定性規則組合。目前缺少獨立的 Safety State 真值與 STM32 上板、延遲、記憶體或量化驗證。
