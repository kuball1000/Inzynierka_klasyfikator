import argparse, json, os, numpy as np
from statistics import mean
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import classifier_voting
# Reuse utility functions from classifier3 (or similar)
from classifier3 import _file_sha1, DEFAULT_INTENTS_JSON

def norm_ep(ep: str) -> str:
    """Ujednolica endpointy (usuwa początkowe /)."""
    ep = (ep or "").strip()
    if ep.startswith("/"):
        ep = ep[1:]
    return ep

def load_testset(path: str):
    """Wczytuje testset JSON (endpoint -> [frazy])."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nie znaleziono pliku: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Plik testset JSON musi być słownikiem endpoint -> [lista tekstów].")
    norm_data = {}
    for ep, samples in data.items():
        nep = norm_ep(ep)
        if not isinstance(samples, list) or not all(isinstance(x, str) for x in samples):
            raise ValueError(f"Wartość dla '{ep}' musi być listą stringów.")
        norm_data[nep] = samples
    if not norm_data:
        raise ValueError("Zbiór testowy jest pusty.")
    return norm_data

def evaluate(intents_path: str, testset_path: str):
    # Load resources for voting classifier
    print("Loading Voting Classifier resources...")
    res = classifier_voting.load_resources(intents_path)
    
    testset = load_testset(testset_path)

    y_true, y_pred, samples_debug = [], [], []
    # Make sure we cover all intent labels
    labels_for_report = sorted(set(list(testset.keys()) + list(res.intents.keys())))

    print(f"Uruchamianie ewaluacji VOTING CLASSIFIER (C1 + C3)")

    for true_ep, texts in testset.items():
        for txt in texts:
            out = classifier_voting.predict_voting(txt, res)
            pred_ep = norm_ep(out.get("endpoint", ""))

            y_true.append(true_ep)
            y_pred.append(pred_ep)
            samples_debug.append({
                "text": txt,
                "true": true_ep,
                "pred": pred_ep,
                "method": out.get("method"),
                "score": out.get("score"),
                "details": out.get("details"), # Extra details for debugging
            })

    acc = accuracy_score(y_true, y_pred)
    pr_micro, rc_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_for_report, average="micro", zero_division=0
    )
    pr_macro, rc_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_for_report, average="macro", zero_division=0
    )

    print("=== WYNIKI OGÓLNE ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (micro): {pr_micro:.4f} | Recall (micro): {rc_micro:.4f} | F1 (micro): {f1_micro:.4f}")
    print(f"Precision (macro): {pr_macro:.4f} | Recall (macro): {rc_macro:.4f} | F1 (macro): {f1_macro:.4f}")
    print()

    print("=== RAPORT PER-ENDPOINT ===")
    print(classification_report(y_true, y_pred, labels=labels_for_report, zero_division=0, digits=4))
    print()

    print("=== MACIERZ POMYŁEK ===")
    cm = confusion_matrix(y_true, y_pred, labels=labels_for_report)
    header = ["true\\pred"] + labels_for_report
    row_fmt = "{:>28} " + " ".join(["{:>28}"] * len(labels_for_report))
    print(row_fmt.format(*header))
    for i, row in enumerate(cm):
        print(row_fmt.format(labels_for_report[i], *row))
    print()

    errors = [s for s in samples_debug if s["true"] != s["pred"]]
    correct = [s for s in samples_debug if s["true"] == s["pred"]]
    if errors:
        print("=== STATYSTYKI SCORE ===")
        mean_correct = mean([float(s["score"]) for s in correct if s["score"] is not None]) if correct else 0
        mean_wrong = mean([float(s["score"]) for s in errors if s["score"] is not None]) if errors else 0
        print(f"Średni score poprawnych: {mean_correct:.4f}")
        print(f"Średni score błędnych:   {mean_wrong:.4f}")
        print()

        print("=== PRZYKŁADOWE BŁĘDY (TOP-10 NAJNIŻSZYCH SCORE) ===")
        errors.sort(key=lambda x: (x.get("score") if x.get("score") is not None else -1))
        for e in errors[:10]:
            print(f"- text: {e['text']}\n  true: {e['true']}\n  pred: {e['pred']} (method={e['method']}, score={e['score']})\n")
    else:
        print("Brak błędnych predykcji 🎉")

def main():
    parser = argparse.ArgumentParser(description="Ewaluacja klasyfikatora Voting.")
    parser.add_argument("--intents", type=str, default="intents2.json")
    parser.add_argument("--testset", type=str, required=True)
    args = parser.parse_args()

    evaluate(args.intents, args.testset)

if __name__ == "__main__":
    main()
