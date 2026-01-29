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

from classifier import (
    load_intents_from_json,
    DEFAULT_INTENTS_JSON,
)
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(BASE_DIR, "examples")
API_KEY = os.getenv("AISTUDIO_API_KEY")
COOKIE_KEY = os.getenv("COOKIE_KEY")
MAX_RETRIES = 3


genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')
print(API_KEY)

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
            print(response.text)
            time.sleep(2)  # prewencyjnie
            return response
        except Exception as e:
            if "429" in str(e):
                # print(response.text)
                print(e)
                print(chat)
                print(prompt)
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
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nie znaleziono pliku: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Plik musi być słownikiem endpoint -> [lista obiektów].")

    ep_key = _norm_ep(endpoint)
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


def load_testset_response(path: str):
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
    "email": "",
    "password": ""
    })
    headers = {
    'Content-Type': 'application/json',
    'Cookie': 'ACCESS_TOKEN=' + COOKIE_KEY
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    set_cookie = response.headers.get("Set-Cookie")

    match = re.search(r"ACCESS_TOKEN=([^;]+)", set_cookie)
    if match:
        access_token = match.group(1)
    return access_token

def set_session():
    url = "http://localhost:5173/api/auth/organization/set-active"

    payload = json.dumps({
    "organizationId": "",
    "organizationSlug": ""
    })
    headers = {
    'Content-Type': 'application/json',
    'Cookie': 'beter-auth.session_token=' + COOKIE_KEY
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    set_cookie = response.headers.get("Set-Cookie")

    match = re.search(r"ACCESS_TOKEN=([^;]+)", set_cookie)
    if match:
        access_token = match.group(1)
    return access_token

def get_session():
    url = "http://localhost:5173/api/auth/get-session"

    headers = {}

    response = requests.request("GET", url, headers=headers)

    token =  response.json()["session"]["token"]

    return token


def evaluate(intents_path: str, testset_path: str):
    intents = load_intents_from_json(intents_path)
    
    testset = load_testset_response(testset_path)
    
    known_eps = sorted({norm_ep(k) for k in intents.keys()})
    labels_for_report = sorted(set(list(known_eps) + list(testset.keys())))

    schema_data = load_schema("schema.json")
    y_true, y_pred = [], []
    samples_debug = []
    results = []
    i = 0
    all_prompts = []
    
    for true_ep, pairs in testset.items():
        if norm_ep(true_ep) != "api/wydarzenia/leczenia":
            continue

        for i, (txt, expected_json) in enumerate(pairs, start=1):
            gen_json = {}
            diff = {}
            
            pred_ep = norm_ep(true_ep)
            method = "assumed_perfect"
            score = 1.0
            
            print(f"Przetwarzanie testu {i}: zakładam endpoint {pred_ep}")

            schema_prompt = get_schema_prompt(pred_ep, schema_data)
            INTENTS_RESP_PATH = os.path.join(BASE_DIR, "intents_response2.json")
            
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
            all_prompts.append(full_prompt)
            try:
                chat = model.start_chat()
                response = safe_send(chat, full_prompt)
                response_text = response.text.strip()
                endpoint, clean = extract_endpoint_and_json(response_text)
                
                print(response_text)
                print(full_prompt)


                try:
                    gen_json = json.loads(clean)
                    expected_json = try_fix_json(expected_json)
                    ref_json = json.loads(expected_json)
                    precision, diff = compare_jsons(ref_json, gen_json)
                    if (i-1) % 5 == 0:
                        password = get_session()
                    
                    response_code = 0
                    # try:
                    #     url = f"http://localhost:3001/{pred_ep.lstrip('/')}"
                    #     cookie_token = 'better-auth.session_token=JIw9hFr5K4oa5Wjof6wpmG6An1Zg1HaF.u0ssrzeifkXjel%2BEv5OREM3ZVE7O%2Ba7zYJHHdttyUOc%3D; better-auth.session_data=eyJzZXNzaW9uIjp7InNlc3Npb24iOnsiZXhwaXJlc0F0IjoiMjAyNi0wMi0yM1QyMjoxMDoxNS40NjVaIiwidG9rZW4iOiJKSXc5aEZyNUs0b2E1V2pvZjZ3cG1HNkFuMVpnMUhhRiIsImNyZWF0ZWRBdCI6IjIwMjYtMDEtMjRUMjI6MTA6MTUuNDY1WiIsInVwZGF0ZWRBdCI6IjIwMjYtMDEtMjRUMjI6MTA6MTYuMTQ3WiIsImlwQWRkcmVzcyI6IiIsInVzZXJBZ2VudCI6Ik1vemlsbGEvNS4wIChXaW5kb3dzIE5UIDEwLjA7IFdpbjY0OyB4NjQpIEFwcGxlV2ViS2l0LzUzNy4zNiAoS0hUTUwsIGxpa2UgR2Vja28pIENocm9tZS8xNDQuMC4wLjAgU2FmYXJpLzUzNy4zNiIsInVzZXJJZCI6Imt3MjB2ZjhvR2U3dk1JdERJaHhOOFJXYTJlNWZMems1IiwiYWN0aXZlT3JnYW5pemF0aW9uSWQiOiJlV2VCbFlxUEU0bXV1RnB0ZWo0QkZNTnhSNXI5amlMRiIsImltcGVyc29uYXRlZEJ5IjpudWxsLCJpZCI6ImNUUGh0VFNOVWJ3ZkRHV3hnb2g5ZGh5Q1VRdExHaElsIn0sInVzZXIiOnsibmFtZSI6IkFkbWluIiwiZW1haWwiOiJhZG1pbkBleGFtcGxlLmNvbSIsImVtYWlsVmVyaWZpZWQiOnRydWUsImltYWdlIjpudWxsLCJjcmVhdGVkQXQiOiIyMDI2LTAxLTI0VDIxOjEzOjUyLjYxNFoiLCJ1cGRhdGVkQXQiOiIyMDI2LTAxLTI0VDIxOjEzOjUzLjAxOVoiLCJyb2xlIjoiYWRtaW4iLCJiYW5uZWQiOmZhbHNlLCJiYW5SZWFzb24iOm51bGwsImJhbkV4cGlyZXMiOm51bGwsImlkIjoia3cyMHZmOG9HZTd2TUl0REloeE44UldhMmU1Zkx6azUifSwidXBkYXRlZEF0IjoxNzY5MjkzMzU2OTU2LCJ2ZXJzaW9uIjoiMSJ9LCJleHBpcmVzQXQiOjE3NjkyOTM2NTY5NTYsInNpZ25hdHVyZSI6IjlJc25EU1RDTVE3NTdrSTRzZ3RYbXd3bVVWZlBfeURBbGxOd083ZktzN2cifQ'
                    #     print(cookie_token)

                    #     headers_json = {
                    #         "accept": "application/json",
                    #         "Content-Type": "application/json",
                    #         "Cookie": cookie_token,
                    #     }
                    #     headers_form = {
                    #         "accept": "application/json",
                    #         "Cookie": cookie_token,
                    #     }

                    #     if pred_ep == "api/konie":
                    #         # przygotuj dane tekstowe (null -> "", wartości jako str)
                    #         form_values = {}
                    #         for k, v in gen_json.items():
                    #             if k == "file" and (v is False or v is None):
                    #                 continue
                    #             form_values[k] = "" if v is None else str(v)

                    #         multipart = {k: (None, v) for k, v in form_values.items()}

                    #         # wyślij multipart/form-data poprawnie
                    #         response = requests.post(url, headers=headers_form, files=multipart)

                    #     else:
                    #         response = requests.post(url, headers=headers_json, json=gen_json)

                    #     response_code = response.status_code
                    #     if response.status_code >= 300:
                    #         print(f"⚠️ Błąd odpowiedzi: {response.text}")
                    # except Exception as curl_error:
                    #     print(f"❌ Nie udało się wysłać zapytania CURL: {curl_error}")
                    #     response_code = 999 

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
                    response_code = 0 # lub inna wartość błędu
                    results.append({
                        "i": f"{i}",
                        "prompt": full_prompt,
                        "endpoint": endpoint,
                        "oczekiwany": expected_json,
                        "gemini": response_text,
                        "response_status_code": response_code
                    })
                    print(f"Błąd parsowania: {parse_error}")

            except Exception as e:
                response_code = 0
                results.append({
                    "i": f"{i}",
                    "prompt": full_prompt,
                    "endpoint": endpoint,
                    "oczekiwany": expected_json,
                    "gemini": "BRAK ODPOWIEDZI",
                    "response_status_code": response_code
                })
                print(f"Błąd Gemini/Sieci: {e}")


            y_true.append(true_ep)
            y_pred.append(pred_ep) 
            
            try:
                prec_val = precision
            except:
                prec_val = 0.0

            # if (response_code >= 200 and response_code < 300 and prec_val >= 0.7):
            #     new = 1
            # elif (response_code >= 200 and response_code < 300 and prec_val >= 0.5 and prec_val < 0.7):
            #     new = 0.5
            # else:
            #     new = 0

            skutecznosc = 1 if response_code >= 200 and response_code < 300 else 0

            samples_debug.append({
                "text": txt,
                "true": true_ep,
                "pred": pred_ep,
                "the_same": 1, # Zawsze 1
                "precision": round(prec_val, 2),
                # "efficiency": new,
                "skteczność": skutecznosc,
                "response_code": response_code,
                "method": method,
                "score": score,
                "gemini": gen_json,
                "diff": diff,
                # "debug": out.get("debug"),
            })

            output_path = Path(__file__).with_name("samples_debug_gemini.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(samples_debug, f, ensure_ascii=False, indent=2)
            
            i += 1

    acc = accuracy_score(y_true, y_pred)
    print("=== WYNIKI GEMINI (przy założeniu idealnego klasyfikatora) ===")
    print(f"Accuracy (klasyfikacji): {acc:.4f} (powinno być 1.0)")

    
    prompts_output_path = Path(__file__).with_name("prompts_log.json")
    with open(prompts_output_path, "w", encoding="utf-8") as f:
        json.dump(all_prompts, f, ensure_ascii=False, indent=2)
    print(f"Zapisano wszystkie prompty do: {prompts_output_path}")

def main():
    parser = argparse.ArgumentParser(description="Ewaluacja Gemini na zbiorze testowym (zakładając poprawny endpoint).")
    parser.add_argument("--intents", type=str, default=DEFAULT_INTENTS_JSON, help="Ścieżka do pliku intents JSON.")
    parser.add_argument("--testset", type=str, required=True, help="Ścieżka do pliku testowego JSON.")
    args = parser.parse_args()

    if not os.path.exists(args.intents):
        raise FileNotFoundError(f"Nie znaleziono pliku intents: {args.intents}")
    if not os.path.exists(args.testset):
        raise FileNotFoundError(f"Nie znaleziono pliku testset: {args.testset}")

    # print(get_session())
    # exit()

    evaluate(args.intents, args.testset)

if __name__ == "__main__":
    main()
