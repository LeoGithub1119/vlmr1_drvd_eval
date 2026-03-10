# GRPO 訓練參數設定與最佳 Checkpoint 選擇指南

這份文件旨在說明如何修改 sbatch 腳本以進行完整的 Epoch 訓練，並提供關於如何選擇最佳 GRPO 訓練週期的建議。

---

### 1. 如何設定以跑完一個完整的 Epoch？

要讓模型完整地看過一次包含 40960 筆資料的數據集，您需要基於目前的設定來計算總共需要多少訓練步數 (`steps`)。

#### 參數：
- **資料集總數 (Dataset Size)**: 40960
- **每張 GPU 的批次大小 (per_device_train_batch_size)**: 1
- **GPU 數量 (num_gpus)**: 4
- **梯度累積步數 (gradient_accumulation_steps)**: 2

#### 計算方式：
1.  **全局批次大小 (Global Batch Size)**:
    `per_device_train_batch_size` × `num_gpus` = 1 × 4 = 4
    *這代表硬體在每一次前向傳播 (forward pass) 中，總共會處理 4 筆資料。*

2.  **有效批次大小 (Effective Batch Size)**:
    `Global Batch Size` × `gradient_accumulation_steps` = 4 × 2 = 8
    *這代表模型每更新一次權重 (optimizer step)，等同於看過了 8 筆資料。*

3.  **一個 Epoch 所需的 Step 數**:
    `資料集總數` / `有效批次大小` = 40960 / 8 = **5120**

#### 如何修改 `full_grpo.sh`：
您需要將 `--max_steps` 參數從 `500` 改為 `5120`。

```bash
# ... (其他參數)
  --max_steps 5120 \
# ... (其他參數)
```

---

### 2. GRPO 需要跑多少個 Epoch 才能找到最佳 Checkpoint？

您的想法「觀察到哪個 EPOCH 之間 LOSS 開始停止進步，取這 N-1 步為最佳 CKPT」是完全正確的，這就是 **Early Stopping** 的核心思想。

對於 GRPO 這類對齊（Alignment）微調任務，通常**不需要**像傳統預訓練那樣跑非常多的 Epochs。過多的訓練很容易導致**過擬合 (Overfitting)**。

#### 建議的 Epoch 數量：
-   **10 個以上的 Epochs 通常太多了**：對於使用 LoRA 的微調任務，超過 10 個 Epochs 幾乎肯定會導致模型性能下降，產生不自然或重複的輸出。
-   **合理的起點是 1-3 個 Epochs**：建議您可以先以 1 個 Epoch (`max_steps=5120`) 為目標進行一次完整的訓練。如果訓練結束時，從監控指標上看仍有進步空間，可以考慮增加到 2 或 3 個 Epochs。

#### 如何判斷最佳 Checkpoint：
在訓練過程中，您應該在 `wandb` (或其他監控工具) 上密切關注以下幾個關鍵指標：

1.  **`train/reward`**: **這是最重要的指標**。在 GRPO 中，我們的目標是最大化這個獎勵分數。理想的曲線是**快速上升，然後進入高原期 (plateau)**，不再有顯著增長。
2.  **`train/loss`**: GRPO 的策略損失。它應該要穩定**下降**，然後同樣進入高原期。
3.  **`train/kl`** (如果 beta > 0): 這個指標衡量您的模型與原始模型的差異程度。如果這個值**持續性地、大幅度地增長**，可能是一個警訊，代表模型為了迎合格式而「忘記」了它原有的語言能力。

**最佳 Checkpoint 的判斷時機**：
當您觀察到 `train/reward` 曲線開始**從快速上升變為平緩**時，那個「轉折點」附近的 checkpoint 通常就是最佳的。如同您所說的，在指標停止進步（進入高原期）之前的那個 checkpoint，往往是泛化能力最好的。

**策略總結：**
1.  先將 `max_steps` 設定為 `5120`，完整地跑完一個 Epoch。
2.  觀察 `wandb` 上的 `reward` 和 `loss` 曲線。
3.  如果曲線在 5120 步之前就已經明顯趨平，那麼最佳的 checkpoint 可能就在更早的地方。
4.  如果曲線在 5120 步時仍在顯著上升/下降，那麼您可以考慮延長訓練，例如設定 `max_steps` 為 `10240` (2 epochs)。
