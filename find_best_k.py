import argparse, json, os
from sklearn.metrics import accuracy_score, f1_score
from classifier2 import build_or_load_index, predict_endpoint, load_intents_from_json, _file_sha1, DEFAULT_INTENTS_JSON

def norm_ep(ep: str) -> str:
    ep = (ep or "").strip()
    if ep.startswith("/"):
        ep = ep[1:]
    return ep

def load_testset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    norm_data = {}
    for ep, samples in data.items():
        nep = norm_ep(ep)
        norm_data[nep] = samples
    return norm_data

def evaluate_k(k: int, idx, intents, testset) -> dict:
    y_true, y_pred = [], []
    for true_ep, texts in testset.items():
        for txt in texts:
            out = predict_endpoint(txt, idx, intents=intents, k_neighbors=k)
            pred_ep = norm_ep(out.get("endpoint", ""))
            y_true.append(true_ep)
            y_pred.append(pred_ep)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"k": k, "acc": acc, "f1_macro": f1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--intents", type=str, default=DEFAULT_INTENTS_JSON)
    parser.add_argument("--testset", type=str, default="testset2.json")
    args = parser.parse_args()

    intents = load_intents_from_json(args.intents)
    intents_hash = _file_sha1(args.intents)
    idx = build_or_load_index(intents, intents_hash)
    testset = load_testset(args.testset)

    print("Evaluating K from 2 to 20...")
    results = []
    for k in range(2, 21):
        res = evaluate_k(k, idx, intents, testset)
        results.append(res)
        print(f"K={k:2d} -> Accuracy: {res['acc']:.4f} | F1 Macro: {res['f1_macro']:.4f}")

    # Sort by Accuracy desc, then F1 desc
    best_results = sorted(results, key=lambda x: (x["acc"], x["f1_macro"]), reverse=True)
    
    print("\nTop 5 K configurations:")
    for i, res in enumerate(best_results[:5]):
        print(f"{i+1}. K={res['k']} (Acc: {res['acc']:.4f}, F1: {res['f1_macro']:.4f})")

    top_5_k = [r["k"] for r in best_results[:5]]
    print(f"\nRecommended Ensemble K list: {sorted(top_5_k)}")

if __name__ == "__main__":
    main()
