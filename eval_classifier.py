# -*- coding: utf-8 -*-
"""
Ewaluacja klasyfikatora intencji -> endpointów API na zbiorze testowym (JSON),
zgodna z classifier.py, który ładuje intents z pliku JSON i cache'uje embeddings
po hash'u pliku intents.

Użycie:
    python eval_classifier.py --intents intents.json --testset testset.json [--rebuild]
"""

import argparse
import json
import os
from typing import Dict, List
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# Import z Twojego classifier.py
from classifier import (
    build_or_load_index,
    predict_endpoint,
    load_intents_from_json,
    _file_sha1,
    DEFAULT_INTENTS_JSON,
)

def norm_ep(ep: str) -> str:
    """Ujednolicenie ścieżek endpointów (np. '/api/konie' -> 'api/konie')."""
    ep = (ep or "").strip()
    if ep.startswith("/"):
        ep = ep[1:]
    return ep

def load_testset(path: str) -> Dict[str, List[str]]:
    """
    Oczekiwany format JSON:
    {
      "api/konie": ["...", "...", ...],
      "api/wydarzenia/leczenia": ["...", ...],
      ...
    }
    """
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

def evaluate(intents_path: str, testset_path: str, force_rebuild: bool = False):
    # 1) Wczytaj intents i policz hash pliku
    intents = load_intents_from_json(intents_path)
    intents_hash = _file_sha1(intents_path)

    # 2) Zbuduj/załaduj indeks (zależny od hash'a intents)
    idx = build_or_load_index(intents=intents, intents_hash=intents_hash, force_rebuild=force_rebuild)

    # 3) Wczytaj zbiór testowy
    testset = load_testset(testset_path)

    # 4) Zbierz listę etykiet (porządek stabilny)
    known_eps = sorted({norm_ep(k) for k in intents.keys()})
    labels_for_report = sorted(set(list(known_eps) + list(testset.keys())))

    # 5) Klasyfikacja
    y_true, y_pred = [], []
    samples_debug = []
    for true_ep, texts in testset.items():
        for txt in texts:
            out = predict_endpoint(txt, idx, intents=intents)
            pred_ep = norm_ep(out.get("endpoint", ""))

            y_true.append(true_ep)
            y_pred.append(pred_ep)
            samples_debug.append({
                "text": txt,
                "true": true_ep,
                "pred": pred_ep,
                "method": out.get("method"),
                "score": out.get("score"),
                "debug": out.get("debug"),
            })

    # 6) Metryki globalne
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

    # 7) Raport per-endpoint
    print("=== RAPORT PER-ENDPOINT ===")
    print(classification_report(y_true, y_pred, labels=labels_for_report, zero_division=0, digits=4))
    print()

    # 8) Macierz pomyłek
    cm = confusion_matrix(y_true, y_pred, labels=labels_for_report)
    print("=== MACIERZ POMYŁEK ===")
    header = ["true\\pred"] + labels_for_report
    row_fmt = "{:>28} " + " ".join(["{:>28}"] * len(labels_for_report))
    print(row_fmt.format(*header))
    for i, row in enumerate(cm):
        print(row_fmt.format(labels_for_report[i], *row))
    print()

    # 9) Przykładowe błędy
    print("=== PRZYKŁADOWE BŁĘDY (TOP-10) ===")
    errors = [s for s in samples_debug if s["true"] != s["pred"]]
    # Najpierw te o najniższym score (najmniej pewne)
    errors.sort(key=lambda x: (x.get("score") if x.get("score") is not None else -1))
    for e in errors[:10]:
        print(f"- text: {e['text']}\n  true: {e['true']}\n  pred: {e['pred']} (method={e['method']}, score={e['score']})\n")

def main():
    parser = argparse.ArgumentParser(description="Ewaluacja klasyfikatora na zbiorze testowym (JSON).")
    parser.add_argument("--intents", type=str, default=DEFAULT_INTENTS_JSON, help="Ścieżka do pliku intents JSON.")
    parser.add_argument("--testset", type=str, required=True, help="Ścieżka do pliku testowego JSON.")
    parser.add_argument("--rebuild", action="store_true", help="Przebuduj indeks embeddingów (opcjonalnie).")
    args = parser.parse_args()

    if not os.path.exists(args.intents):
        raise FileNotFoundError(f"Nie znaleziono pliku intents: {args.intents}")
    if not os.path.exists(args.testset):
        raise FileNotFoundError(f"Nie znaleziono pliku testset: {args.testset}")

    evaluate(args.intents, args.testset, force_rebuild=args.rebuild)

if __name__ == "__main__":
    main()
