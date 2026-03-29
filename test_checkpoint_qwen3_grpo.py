
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
import os
"""
這個腳本是一個**定性分析 (Qualitative Analysis)** 和**健全性檢查 (Sanity Check)** 工具，而不是一個定量的評估基準 (Quantitative Benchmark)。它非常適合在訓練初期或中期快速判斷模型是否「走在正確的路上」。

以下我將針對您的三個問題詳細解釋：

### 1. 為何這個腳本適合評估 GRPO 之後的 Model？

GRPO (Group Relative Policy Optimization) 的核心目標，與 DPO 或 RLHF 類似，是讓模型的**輸出行為**更符合人類的偏好或特定的格式要求。在您的案例中，這個「偏好」就是一個**極度嚴格的輸出格式**。

因此，評估的重點**不僅僅是答案對不對**，更重要的是**模型有沒有學會按照您規定的格式來回答**。

這個腳本之所以適合，是因為它直接模擬了訓練時的場景：
*   **輸入一致性**：它提供給模型的 Prompt 結構（包含圖片、問題、選項、以及詳細的格式指令）與訓練資料 `grpo_qa_list_fixed.jsonl` 中的人類提問 (`human` turn) 完全一致。
*   **目標導向**：它直接測試了 GRPO 訓練的核心目標——「格式遵循能力」。如果模型連這個都做不到，那麼答案的準確率就沒有意義了。

### 2. 它會做什麼樣的評估來判斷？

這個腳本執行了兩個層級的評估：

**層級一：自動化的結構檢查 (Automated Structural Check)**

這是腳本的核心功能。在生成模型的輸出後，它會用簡單的字串檢查來判斷：

1.  **是否存在 `<think>...</think>` 區塊**：檢查輸出中是否同時包含開始標籤 `<think>` 和結束標籤 `</think>`。
2.  **是否存在 `<answer>...</answer>` 區塊**：同上，檢查答案區塊的完整性。
3.  **是否存在 `<location>...</location>` 區塊**：同上，檢查位置區塊的完整性。

這是一個快速、自動化的「及格線」檢查。只要有任何一個區塊缺失，驗證就會顯示 `❌ Missing`。

**層級二：提供給人類的內容審查 (Content Review for Humans)**

腳本會將完整的「Parsed Output」印在終端機上。這讓您可以進行更深入的、機器難以判斷的評估：

1.  **內容的邏輯性**：`<obs>` 的描述是否屬實？`<evidence>` 提出的證據是否合理？`<logic>` 的推論是否連貫？
2.  **答案的正確性**：`<answer>` 裡的選項（A, B, C）是否正確？
3.  **位置的準確性**：`<location>` 標示的數字是否真的對應到瑕疵的位置？如果沒有瑕疵，它是否正確地輸出了 `<location></location>`？
4.  **無冗餘輸出**：模型是否在規定的三個標籤之外，生成了任何多餘的文字（例如 "好的，這是一個..."）？

### 3. 判斷的依據又是什麼？

判斷的依據完全來自於您在 `full_grpo.sh` 中提供的訓練資料和 Prompt 中定義的**規則**。

*   **主要依據 (結構)**：
    *   腳本的判斷依據是：模型輸出**是否包含**您指定的 `<think>`, `<answer>`, `<location>` 這三個成對的標籤。

*   **次要依據 (內容，由您判斷)**：
    *   `<think>` 內部是否嚴格遵循 `<obs>...</obs><evidence>...</evidence><logic>...</logic>` 的順序和內容要求。
    *   `<answer>` 內的字母是否為 A, B, C, D 其中之一。
    *   `<location>` 內的數字是否為 1-9 的組合，或在無瑕疵時為空。

總結來說，這個腳本是一個**快速驗證工具**，它自動檢查最基本的格式要求，並提供完整的輸出讓您（人類專家）來判斷更細微的內容品質。對於評估一個以「格式遵循」為主要目標的微調任務來說，這是非常有效且必要的第一步。
"""
def main():
    # --- Configuration ---
    base_model_path = "/work/foobarbaz911/vlmr1/models/Qwen3-VL-8B-Instruct"
    # adapter_path = "/home/foobarbaz911/VLM-R1/temp/grpo_qa_list_133473/checkpoint-2"
    adapter_path = "/work/foobarbaz911/vlmr1/outputs/grpo_134897/checkpoint-5120"
    image_root = "/work/foobarbaz911/vlmr1/datasets/IAD256"

    # --- Pick two images for testing ---
    # Using one image as the 'test' image and another as the 'reference'
    test_image_path = os.path.join(image_root, "DS-MVTec/pill/train/good/025.png")
    reference_image_path = os.path.join(image_root, "DS-MVTec/pill/train/good/032.png")

    print(f"Test image: {test_image_path}")
    print(f"Reference image: {reference_image_path}")

    # --- Load Model and Processor ---
    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(adapter_path, trust_remote_code=True)
    
    model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        torch_dtype="auto",
        device_map="cuda",
        trust_remote_code=True,
    )
    
    # Apply the LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    print("Model loaded successfully.")

    # --- Prepare Prompt and Images ---
    # This is a simplified version of the prompt structure from your dataset
    prompt_text = (
        "You are given two images of the same industrial object.\n"
        "- The first image is the test image.\n"
        "- The second image is a corresponding 1-shot normal reference image of the same object.\n\n"
        "The test image may either be normal or contain defects.\n"
        "Your task is to carefully compare the test image with the normal reference image, determine whether any defect exists, identify its location if present, and choose the most appropriate answer option.\n\n"
        "<question>\nIs there any defect in the object?\n</question>\n\n"
        "<options>\nA: No.\nB: Yes.\nC: I don't know.\n</options>\n\n"
        "IMPORTANT INSTRUCTIONS:\n"
        "1. You MUST reason step by step INSIDE a <think>...</think> block.\n"
        "2. After reasoning, output EXACTLY ONE final choice INSIDE an <answer>...</answer> block.\n"
        "3. Immediately after <answer>, output defect locations INSIDE a <location>...</location> block.\n"
        "4. If the image is normal and no defect is found, output <location></location>.\n"
        "5. DO NOT write anything outside <think>, <answer>, or <location>.\n\n"
        "REASONING FORMAT (STRICT):\n"
        "Inside <think>, you MUST include the following tags IN THIS EXACT ORDER:\n"
        "<obs>...</obs><evidence>...</evidence><logic>...</logic>\n\n"
        "Now respond using the following format ONLY:\n"
        "<think><obs>...</obs><evidence>...</evidence><logic>...</logic></think><answer>X</answer><location>...</location>"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    images = [Image.open(test_image_path), Image.open(reference_image_path)]
    
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    inputs = processor(text=text, images=images, return_tensors="pt").to("cuda")

    # --- Generate Output ---
    print("\n--- Generating Model Output ---")
    generation_kwargs = {
        "max_new_tokens": 512,
        "do_sample": False,
    }
    
    with torch.no_grad():
        res = model.generate(**inputs, **generation_kwargs)

    response = processor.batch_decode(res, skip_special_tokens=False)[0]

    # Clean up the response to only show the generated part
    try:
        # The response includes the prompt, so we find where the prompt ends and show the rest
        output = response.split("<|im_start|>assistant\n")[1].replace("<|im_end|>", "").strip()
    except IndexError:
        output = response # Fallback if split fails

    print("\n--- Raw Response ---")
    print(response)
    print("\n--- Parsed Output ---")
    print(output)
    print("---------------------\n")

    # --- Verification Checks ---
    print("--- Verification ---")
    if "<think>" in output and "</think>" in output:
        print("✅ <think>...</think> block found.")
    else:
        print("❌ Missing or incomplete <think>...</think> block.")

    if "<answer>" in output and "</answer>" in output:
        print("✅ <answer>...</answer> block found.")
    else:
        print("❌ Missing or incomplete <answer>...</answer> block.")

    if "<location>" in output and "</location>" in output:
        print("✅ <location>...</location> block found.")
    else:
        print("❌ Missing or incomplete <location>...</location> block.")

if __name__ == "__main__":
    main()
