import json

with open("judge_results.json") as f:
    data = json.load(f)

def compute(name, arr):
    A = arr.count("A")
    B = arr.count("B")
    T = arr.count("TIE")
    total = len(arr)

    print(f"\n{name}")
    print("A win rate:", A/total)
    print("B win rate:", B/total)
    print("Tie rate:", T/total)

compute("Base vs Stage1", data["base_vs_stage1"])
compute("Stage1 vs Stage2", data["stage1_vs_stage2"])

# 
compute("Stage2 vs Epoch1", data["stage2_vs_epoch1"])
compute("Stage2 vs LR1e5", data["stage2_vs_lr1e5"])