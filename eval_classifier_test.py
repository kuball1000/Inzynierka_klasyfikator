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
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from pathlib import Path
import random
import csv
import google.generativeai as genai
import time
import re
import requests
from dotenv import load_dotenv

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
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(BASE_DIR, "examples")
API_KEY = os.getenv("AISTUDIO_API_KEY")
COOKIE_KEY = os.getenv("COOKIE_KEY")
MAX_RETRIES = 3


genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

file_to_endpoint = {
    "examples_konie.tsv": "api/konie",
    "Inżynierka-choroby.tsv": "api/wydarzenia/choroby",
    "Inżynierka-kowale.tsv": "api/kowale",
    "Inżynierka-podkucia.tsv": "api/wydarzenia/podkucie",
    "Inżynierka-rozrody.tsv": "api/wydarzenia/rozrody",
    "Inżynierka-weterynarze.tsv": "api/weterynarze",
    "Inżynierka-wydarzenia_profilaktyczne.tsv": "api/wydarzenia/zdarzenia_profilaktyczne"
}

endpoint_to_file = {v: k for k, v in file_to_endpoint.items()}

def norm_ep(ep: str) -> str:
    """Ujednolicenie ścieżek endpointów (np. '/api/konie' -> 'api/konie')."""
    ep = (ep or "").strip()
    if ep.startswith("/"):
        ep = ep[1:]
    return ep

def _norm_ep(ep: str) -> str:
    ep = ep.strip()
    ep = re.sub(r"^/+", "", ep)
    ep = re.sub(r"/{2,}", "/", ep)
    return ep

def safe_send(chat, prompt):
    for i in range(MAX_RETRIES):
        try:
            response = chat.send_message(prompt)
            time.sleep(2)  # prewencyjnie
            return response
        except Exception as e:
            if "429" in str(e):
                print("⏳ Przekroczony limit — czekam 30s...")
                time.sleep(30)
            else:
                raise e
    raise RuntimeError("Zbyt wiele błędów 429 — przerywam.")

def try_fix_json(text):
    text = text.strip()

    if text.startswith("```json"):
        text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("```"))

    text = text.replace('""', '"')
    text = re.sub(r'"([a-zA-Z0-9_]+)"\s*:\s*,\s*"([^"]+)"', r'"\1": "\2"', text)
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r',\s*}', '}', text)
    text = text.encode("utf-8", "ignore").decode("utf-8").replace('\ufeff', '')

    return text

def extract_endpoint_and_json(text):
    text = extract_json_block(text)
    endpoint_match = re.search(r"Endpoint:\s*`?(/?api/[^\s`]+)`?", text)
    endpoint = endpoint_match.group(1).strip() if endpoint_match else "NIEZNANY"
    cleaned_text = re.sub(r"Endpoint:\s*`?(/?api/[^\s`]+)`?", "", text).strip()

    return endpoint, cleaned_text

def _join_desc_info(description: str, informations: str) -> str:
    description = (description or "").strip()
    informations = (informations or "").strip()
    return description if not informations else f"{description}\n{informations}"


def load_all_examples_for_endpoint(
    path: str,
    endpoint: str,
    decode_output: bool = False
) -> List[Tuple[str, Any]]:
    """
    Czyta intents_reponse.json (format: endpoint -> [{description, informations, output}, ...])
    i zwraca WSZYSTKIE przykłady dla danego endpointu jako listę:
      (prompt_text, output_text_lub_dict).

    - prompt_text = description + (opcjonalnie) '\n' + informations
    - jeśli decode_output=True, to output zwracany jako dict (json.loads)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nie znaleziono pliku: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Plik musi być słownikiem endpoint -> [lista obiektów].")

    ep_key = _norm_ep(endpoint)
    # spróbuj dopasować klucz po normalizacji
    candidates: Dict[str, str] = { _norm_ep(k): k for k in data.keys() }
    if ep_key not in candidates:
        available = ", ".join(sorted(candidates.keys()))
        raise KeyError(f"Endpoint '{endpoint}' nie znaleziony. Dostępne: {available}")

    raw_list = data[candidates[ep_key]]
    if not isinstance(raw_list, list):
        raise ValueError(f"Wartość dla '{endpoint}' musi być listą obiektów.")

    out: List[Tuple[str, Any]] = []
    for i, item in enumerate(raw_list):
        if not isinstance(item, dict):
            raise ValueError(f"[{endpoint}][{i}] musi być obiektem z polami description/informations/output.")
        for k in ("description", "informations", "output"):
            if k not in item or not isinstance(item[k], str):
                raise ValueError(f"[{endpoint}][{i}] pole '{k}' musi istnieć i być stringiem.")

        prompt = _join_desc_info(item["description"], item["informations"])

        # walidacja i ewentualne dekodowanie output
        try:
            parsed = json.loads(item["output"])
        except Exception as e:
            raise ValueError(f"[{endpoint}][{i}] 'output' nie jest poprawnym JSON-em: {e}")

        out.append((prompt, parsed if decode_output else item["output"]))

    return out

def extract_json_block(text):
    if text.startswith("```"):
        return "\n".join(line for line in text.splitlines() if not line.strip().startswith("```")).strip()
    return text.strip()

def compare_jsons(expected, generated):
    expected_keys = set(expected.keys())
    generated_keys = set(generated.keys())

    all_keys = expected_keys | generated_keys
    matched = 0
    differences = {}

    for key in all_keys:
        val1 = expected.get(key)
        val2 = generated.get(key)
        if val1 == val2:
            matched += 1
        else:
            differences[key] = {"expected": val1, "generated": val2}

    precision = matched / len(all_keys) if all_keys else 1.0
    return precision, differences


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

def load_testset_response(path: str):
    """
    Ładuje testset_response.json w formacie:
    {
      "api/konie": [
        { "description": "...", "informations": "...", "output": "{...}" },
        ...
      ],
      ...
    }

    Zwraca:
      dict(endpoint -> list[tuple(description_text, output_json_text)])
    """
    import json, os, re

    def norm_ep(ep: str) -> str:
        ep = ep.strip()
        ep = re.sub(r"^/+", "", ep)
        ep = re.sub(r"/{2,}", "/", ep)
        return ep

    if not os.path.exists(path):
        raise FileNotFoundError(f"Nie znaleziono pliku: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Plik testset_response.json musi być słownikiem endpoint -> lista obiektów.")

    norm_data = {}
    for ep, samples in data.items():
        nep = norm_ep(ep)
        if not isinstance(samples, list):
            raise ValueError(f"Wartość dla '{ep}' musi być listą obiektów.")
        pairs = []
        for i, item in enumerate(samples):
            if not isinstance(item, dict):
                raise ValueError(f"[{ep}][{i}] musi być obiektem z polami description/informations/output.")
            desc = str(item.get("description", "")).strip()
            info = str(item.get("informations", "")).strip()
            out = str(item.get("output", "")).strip()
            if not desc or not out:
                continue
            text = f"{desc}\n{info}" if info else desc
            pairs.append((text, out))
        norm_data[nep] = pairs
    return norm_data



def _load_testset_response(path: str) -> Dict[str, List[str]]:
    """
    Oczekiwany format JSON:
    {
      "api/konie": [
        { "description": "...", "informations": "...", "output": "{...}" },
        ...
      ],
      "api/wydarzenia/choroby": [
        { "description": "...", "informations": "...", "output": "{...}" },
        ...
      ],
      ...
    }

    Zwraca:
      dict(endpoint -> list[stringów]) – czyli połączony tekst description + informations
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nie znaleziono pliku: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Plik testset JSON musi być słownikiem endpoint -> lista obiektów.")

    def norm_ep(ep: str) -> str:
        ep = ep.strip()
        ep = re.sub(r"^/+", "", ep)
        ep = re.sub(r"/{2,}", "/", ep)
        return ep

    norm_data: Dict[str, List[str]] = {}
    json_data: Dict[str, str] = {}

    for ep, samples in data.items():
        nep = norm_ep(ep)

        if not isinstance(samples, list):
            raise ValueError(f"Wartość dla '{ep}' musi być listą obiektów.")

        out_texts: List[str] = []

        for i, item in enumerate(samples):
            if not isinstance(item, dict):
                raise ValueError(f"[{ep}][{i}] musi być obiektem z polami description/informations/output.")
            for key in ("description", "informations", "output"):
                if key not in item or not isinstance(item[key], str):
                    raise ValueError(f"[{ep}][{i}] pole '{key}' musi istnieć i być stringiem.")

            desc = item["output"].strip()
            _desc = item["description"].strip()
            text = desc

            out_texts.append(text)

            # sprawdź, czy output to poprawny JSON (tylko walidacja)
            try:
                json.loads(item["output"])
            except Exception as e:
                raise ValueError(f"[{ep}][{i}] pole 'output' nie jest poprawnym JSON-em: {e}")

        if not out_texts:
            raise ValueError(f"Lista dla '{ep}' jest pusta.")
        norm_data[nep] = out_texts
        json_data[nep] = _desc

    if not norm_data:
        raise ValueError("Zbiór testowy jest pusty.")

    return norm_data, json_data

def load_schema(path: str = "schema.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_schema_prompt(endpoint: str, schema_data: dict) -> str:
    if endpoint not in schema_data:
        return f"(Brak schematu dla endpointu: {endpoint})"
    
    content = schema_data[endpoint]
    for content_type, data in content.items():
        if "schema" in data:
            schema_prompt = json.dumps(data["schema"], indent=2, ensure_ascii=False)
            return schema_prompt
    return "(Brak sekcji 'schema' w tym endpointzie)"

import requests
import json
def login():
    url = "http://localhost:3001/api/login"

    payload = json.dumps({
    "email": "tomek5@wp.pl",
    "password": "12345678"
    })
    headers = {
    'Content-Type': 'application/json',
    'Cookie': 'ACCESS_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOjMsImV4cCI6MTc2MDQ3MzUwNH0.PHqKviIkTpsRHXwgJaiBTyui1-Z32OabFXUN1Humq9A'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    set_cookie = response.headers.get("Set-Cookie")

    match = re.search(r"ACCESS_TOKEN=([^;]+)", set_cookie)
    if match:
        access_token = match.group(1)
    return access_token


def evaluate(intents_path: str, testset_path: str, force_rebuild: bool = False):
    # 1) Wczytaj intents i policz hash pliku
    intents = load_intents_from_json(intents_path)
    intents_hash = _file_sha1(intents_path)

    # 2) Zbuduj/załaduj indeks (zależny od hash'a intents)
    idx = build_or_load_index(intents=intents, intents_hash=intents_hash, force_rebuild=force_rebuild)

    # 3) Wczytaj zbiór testowy
    testset = load_testset_response(testset_path)
    # exit()

    # 4) Zbierz listę etykiet (porządek stabilny)
    known_eps = sorted({norm_ep(k) for k in intents.keys()})
    labels_for_report = sorted(set(list(known_eps) + list(testset.keys())))

    # 5) Klasyfikacja
    schema_data = load_schema("schema.json")
    y_true, y_pred = [], []
    samples_debug = []
    results = []
    i = 0
    # testset to teraz lista stringów (tylko description)
    for true_ep, pairs in testset.items():
        for i, (txt, expected_json) in enumerate(pairs, start=1):
            # exit()
            out = predict_endpoint(txt, idx, intents=intents)
            pred_ep = norm_ep(out.get("endpoint", ""))
            # print(f"Przetwarzanie testu {i}: przewidziano {pred_ep}, oczekiwano {true_ep}")
            schema_prompt = get_schema_prompt(pred_ep, schema_data)
            INTENTS_RESP_PATH = os.path.join(BASE_DIR, "intents_response.json")
            examples_parsed = load_all_examples_for_endpoint(INTENTS_RESP_PATH, pred_ep, decode_output=True)


            full_prompt = f"""Schemat danych wejściowych dla {pred_ep} (format JSON):
                {schema_prompt}

                Twoim zadaniem jest wygenerować poprawny obiekt JSON na podstawie opisu użytkownika. 
                Oto przykłady treningowe, które pomogą Ci zrozumieć, jak powinien wyglądać format JSON odpowiedzi:

            """
            for train_prompt, train_json in examples_parsed:
                full_prompt += f""""
                    Użytkownik: {train_prompt}
                    Odpowiedź JSON:
                    {train_json}\n
                """

            full_prompt += f"""Użytkownik: {txt}
                Odpowiedź JSON:
            """
    # request przeszedł i precyzja > 0.7 to efektywność 1; if request przeszedł i przecyzja [0.5, 0.7] to efektywność 0.5; else 0
            try:
                chat = model.start_chat()
                response = safe_send(chat, full_prompt)
                response_text = response.text.strip()
                endpoint, clean = extract_endpoint_and_json(response_text)
                # print(full_prompt)
                
                # print("\n")
                print(response_text)


                try:
                    gen_json = json.loads(clean)
                    expected_json = try_fix_json(expected_json)
                    ref_json = json.loads(expected_json)
                    precision, diff = compare_jsons(ref_json, gen_json)
                    if (i-1) % 5 == 0:
                        password = login()
                    # exit()

                    try:
                        url = f"http://localhost:3001/{pred_ep.lstrip('/')}"
                        cookie_token = f"ACCESS_TOKEN={password}"
                        print(cookie_token)

                        headers_json = {
                            "accept": "application/json",
                            "Content-Type": "application/json",
                            "Cookie": cookie_token,
                        }
                        headers_form = {
                            "accept": "application/json",
                            "Cookie": cookie_token,
                        }

                        if pred_ep == "api/konie":
                            # przygotuj dane tekstowe (null -> "", wartości jako str)
                            form_values = {}
                            for k, v in gen_json.items():
                                if k == "file" and (v is False or v is None):
                                    continue
                                form_values[k] = "" if v is None else str(v)

                            multipart = {k: (None, v) for k, v in form_values.items()}

                            curl_form_parts = " \\\n  ".join(
                                [f"--form '{k}={v}'" for k, v in form_values.items()]
                            )
                            curl_command = f"""curl --location '{url}' \\
                        --header 'accept: application/json' \\
                        --header 'Cookie: {cookie_token}' \\
                        {curl_form_parts}"""

                            # print("📤 CURL (form-data):\n")
                            # print(curl_command)
                            # print("-" * 80)

                            # wyślij multipart/form-data poprawnie
                            response = requests.post(url, headers=headers_form, files=multipart)

                        else:
                            curl_json_string = json.dumps(gen_json, ensure_ascii=False, indent=2)
                            curl_command = f"""curl --location '{url}' \\
                        --header 'accept: application/json' \\
                        --header 'Content-Type: application/json' \\
                        --header 'Cookie: {cookie_token}' \\
                        --data '{curl_json_string}'"""

                            # print("📤 CURL (JSON):\n")
                            # print(curl_command)
                            # print("-" * 80)

                            response = requests.post(url, headers=headers_json, json=gen_json)

                        response_code = response.status_code
                        # print(f"🌐 [TEST {i}] Wysłano do {endpoint} — kod: {response.status_code}")
                        if response.status_code >= 300:
                            print(f"⚠️ Błąd odpowiedzi: {response.text}")
                    except Exception as curl_error:
                        print(f"❌ Nie udało się wysłać zapytania CURL: {curl_error}")

                    results.append({
                        "i": f"{i}",
                        "prompt": full_prompt,
                        "endpoint": endpoint,
                        "oczekiwany": json.dumps(ref_json, ensure_ascii=False),
                        "gemini": json.dumps(gen_json, ensure_ascii=False),
                        "różnica": json.dumps(diff, ensure_ascii=False),
                        "precyzja": round(precision, 2),
                        "response_status_code": response_code
                    })


                except Exception as parse_error:
                    results.append({
                        "i": f"{i}",
                        "prompt": full_prompt,
                        "endpoint": endpoint,
                        "oczekiwany": expected_json,
                        "gemini": response_text,
                        # "różnica": f"Błąd parsowania JSON: {str(parse_error)}",
                        # "precyzja": 0.0,
                        "response_status_code": response_code
                    })

            except Exception as e:
                results.append({
                    "i": f"{i}",
                    "prompt": full_prompt,
                    "endpoint": endpoint,
                    "oczekiwany": expected_json,
                    "gemini": "BRAK ODPOWIEDZI",
                    "response_status_code": response_code
                })


            y_true.append(true_ep)
            y_pred.append(pred_ep)

            if (response_code >= 200 and response_code < 300 and precision >= 0.7):
                new = 1
            elif (response_code >= 200 and response_code < 300 and precision >= 0.5 and precision < 0.7):
                new = 0.5
            else:
                new = 0

            skutecznosc = 1 if response_code >= 200 and response_code < 300 else 0

            samples_debug.append({
                "text": txt,
                "true": true_ep,
                "pred": pred_ep,
                "the_same": 1 if true_ep == pred_ep else 0,
                "precision": round(precision, 2),
                "efficiency": new,
                "skteczność": skutecznosc,
                "response_code": response_code,
                "method": out.get("method"),
                "score": out.get("score"),
                "debug": out.get("debug"),
            })
            # print(results)
            output_path = Path(__file__).with_name("samples_debug.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(samples_debug, f, ensure_ascii=False, indent=2)
            print(samples_debug)
            i += 1
            # exit()
    print(samples_debug)
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
