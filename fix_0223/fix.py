import json, re, sys
from pathlib import Path

LETTER_RE = re.compile(r"\b([A-H])\b", re.IGNORECASE)

def extract_from_assistant_tail(raw: str) -> str:
    """
    從 raw_model_output 抽出最後一段 assistant 的作答字母。
    你資料常見格式：... \nassistant\nC
    """
    if not raw or not isinstance(raw, str):
        return ""

    # 1) 優先抓最後一次出現 "assistant" 之後的內容
    #    兼容 \nassistant\n 或 "assistant\n"
    parts = re.split(r"\nassistant\n|\bassistant\s*\n", raw, flags=re.IGNORECASE)
    tail = parts[-1] if parts else raw

    # 2) 在 tail 裡找第一個 A-H（通常就是 C / D 這種）
    m = LETTER_RE.search(tail.strip())
    if m:
        return m.group(1).upper()

    # 3) 如果 tail 找不到（少數），退而求其次：找全文最後一個 A-H
    allm = LETTER_RE.findall(raw)
    return allm[-1].upper() if allm else ""

def fix_jsonl(in_path: str, out_path: str):
    in_path = Path(in_path)
    out_path = Path(out_path)
    n = 0
    fixed = 0
    empty = 0

    with in_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # 你檔案裡常見欄位名：raw_model_output
            # 若你實際是其他名字（如 "model_output"），可以在這裡加進去
            raw = obj.get("raw_model_output") or obj.get("model_output") or obj.get("output") or ""

            ans = extract_from_assistant_tail(raw)
            n += 1

            if ans:
                obj["model_response"] = ans
                obj["_raw_pred_text"] = ans
                fixed += 1
            else:
                empty += 1
                # 保留原樣，但也標記一下方便你抽查
                obj["_postfix_note"] = "no_answer_extracted_from_raw_model_output"

            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[OK] {in_path.name} -> {out_path.name}")
    print(f"     total={n}, fixed={fixed}, empty={empty}")

if __name__ == "__main__":
    # 用法：
    # python fix_from_raw.py /path/in.jsonl /path/out.fixed.jsonl
    fix_jsonl(sys.argv[1], sys.argv[2])