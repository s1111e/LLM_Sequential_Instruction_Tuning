import json
import time
from openai import OpenAI

# =========================
# CONFIG (UTSA MODEL)
# =========================
client = OpenAI(
    base_url="http://10.246.100.230/v1",
    api_key="gpustack_50e00c9281422bc5_0c0696dfcb1696d7635e58a2e56d6282"
)

MODEL = "llama-3.3-70b-instruct-awq"


# =========================
# JUDGE FUNCTION
# =========================
def judge(prompt, resp_a, resp_b):

    judge_prompt = f"""
You are an expert evaluator.

Instruction:
{prompt}

Response A:
{resp_a}

Response B:
{resp_b}

Which is better? Answer ONLY:
A or B or TIE
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()

# =========================
# RUN JUDGE
# =========================
def run():

    with open("eval_results.json") as f:
        data = json.load(f)

    base = data["base"]
    s1   = data["stage1"]
    s2   = data["stage2"]

    # 🔥 yeni modeller (varsa)
    s2_epoch1 = data.get("stage2_epoch1")
    s2_lr1e5  = data.get("stage2_lr1e5")

    results = {
        "base_vs_stage1": [],
        "stage1_vs_stage2": [],
        "stage2_vs_epoch1": [],
        "stage2_vs_lr1e5": []
    }

    # -------------------------
    # BASE vs STAGE1
    # -------------------------
    for i in range(len(base)):
        res = judge(
            base[i]["instruction"],
            base[i]["prediction"],
            s1[i]["prediction"]
        )
        results["base_vs_stage1"].append(res)
        print(f"[base vs s1] {i}: {res}")
        time.sleep(1)

    # -------------------------
    # STAGE1 vs STAGE2
    # -------------------------
    for i in range(len(base)):
        res = judge(
            base[i]["instruction"],
            s1[i]["prediction"],
            s2[i]["prediction"]
        )
        results["stage1_vs_stage2"].append(res)
        print(f"[s1 vs s2] {i}: {res}")
        time.sleep(1)

    # -------------------------
    # STAGE2 vs EPOCH1
    # -------------------------
    if s2_epoch1:
        for i in range(len(base)):
            res = judge(
                base[i]["instruction"],
                s2[i]["prediction"],
                s2_epoch1[i]["prediction"]
            )
            results["stage2_vs_epoch1"].append(res)
            print(f"[s2 vs epoch1] {i}: {res}")
            time.sleep(1)

    # -------------------------
    # STAGE2 vs LR1e5
    # -------------------------
    if s2_lr1e5:
        for i in range(len(base)):
            res = judge(
                base[i]["instruction"],
                s2[i]["prediction"],
                s2_lr1e5[i]["prediction"]
            )
            results["stage2_vs_lr1e5"].append(res)
            print(f"[s2 vs lr1e5] {i}: {res}")
            time.sleep(1)

    with open("judge_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✅ Judge DONE")


if __name__ == "__main__":
    run()