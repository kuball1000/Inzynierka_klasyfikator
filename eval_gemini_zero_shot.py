import argparse
import json
import os
from typing import Dict, List, Tuple, Any
from pathlib import Path
import google.generativeai as genai
import time
import re
import requests
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
API_KEY = os.getenv("AISTUDIO_API_KEY")
COOKIE_KEY = os.getenv("COOKIE_KEY")
MAX_RETRIES = 3

if not API_KEY:
    print("WARNING: AISTUDIO_API_KEY not found in .env")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

DEFAULT_INTENTS_JSON = "intents.json"

def norm_ep(ep: str) -> str:
    ep = (ep or "").strip()
    if ep.startswith("/"):
        ep = ep[1:]
    return ep

def safe_send(chat, prompt):
    for i in range(MAX_RETRIES):
        try:
            response = chat.send_message(prompt)
            print(f"--- Gemini Response (try {i+1}) ---")
            # print(response.text[:200] + "..." if len(response.text)>200 else response.text)
            time.sleep(2)
            return response
        except Exception as e:
            if "429" in str(e):
                print("⏳ 429 Limit - waiting 30s...")
                time.sleep(30)
            else:
                raise e
    raise RuntimeError("Too many 429 errors.")

def extract_json_block(text):
    if text.startswith("```"):
        return "\n".join(line for line in text.splitlines() if not line.strip().startswith("```")).strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def try_fix_json(text):
    text = text.strip()
    if text.startswith("```json"):
        text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("```"))
    
    text = text.replace('""', '"')
    text = re.sub(r'"([a-zA-Z0-9_]+)"\s*:\s*,\s*"([^"]+)"', r'"\1": "\2"', text)
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r',\s*}', '}', text)
    # Remove BOM
    text = text.encode("utf-8", "ignore").decode("utf-8").replace('\ufeff', '')
    return text

def extract_endpoint_and_json(text):
    endpoint_match = re.search(r"Endpoint:\s*`?(/?api/[^\s`\n]+)`?", text, re.IGNORECASE)
    endpoint = endpoint_match.group(1).strip() if endpoint_match else "NIEZNANY"
    
    json_text = extract_json_block(text)
    
    json_text = re.sub(r"Endpoint:\s*`?(/?api/[^\s`\n]+)`?", "", json_text, flags=re.IGNORECASE).strip()
    
    return endpoint, json_text

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
    import json
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    norm_data = {}
    for ep, samples in data.items():
        nep = norm_ep(ep)
        pairs = []
        for item in samples:
            desc = str(item.get("description", "")).strip()
            info = str(item.get("informations", "")).strip()
            out = str(item.get("output", "")).strip()
            if not desc or not out: continue
            text = f"{desc}\n{info}" if info else desc
            pairs.append((text, out))
        norm_data[nep] = pairs
    return norm_data

def evaluate(intents_path: str, testset_path: str):
    with open(intents_path, "r", encoding="utf-8") as f:
        intents_data = json.load(f)
    
    with open("schema.json", "r", encoding="utf-8") as f:
        schema_data = json.load(f)
    
    schema_prompt_full = json.dumps(schema_data, indent=2, ensure_ascii=False)
    state_prompt_full = json.dumps(intents_data, indent=2, ensure_ascii=False)
    
    testset = load_testset_response(testset_path)

    y_true = []
    y_pred = []
    samples_debug = []
    all_prompts = []
    i_global = 0
    
    for true_ep, pairs in testset.items():
        true_ep = norm_ep(true_ep)
        
        for txt, expected_json_str in pairs:
            i_global += 1
            print(f"--- Test #{i_global} [True: {true_ep}] ---")
            
            full_prompt = f"""
ROLA:
Jesteś systemem przetwarzania zapytań użytkownika na poprawne struktury JSON
zgodne z interfejsem REST. Twoim zadaniem jest dokładna analiza opisu oraz
wygenerowanie wyłącznie poprawnego obiektu JSON zgodnego ze schematem danych.


ZASADY:
1. Odpowiadaj WYŁĄCZNIE w poprawnym formacie JSON.
2. Nie dodawaj żadnych pól, kluczy ani wartości, które nie występują w schemacie.
3. Jeśli użytkownik opisuje dane niezgodne ze schematem - pomiń je.
4. Jeśli użytkownik podaje wartość niepoprawną typologicznie - dopasuj ją do schematu,
ale nie zgaduj wartości niepodanych.
5. Nie generuj tekstu objaśniającego ani komentarzy.
6. Wybierz najbardziej pasujący endpoint REST na podstawie struktury danych.
7. Analizuj problem krok po kroku (nie wyświetlaj rozumowania) i dopiero potem wygeneruj JSON.
8. Jeśli dane mają odwołanie do istniejących obiektów, używaj wartości ze stanu bazy.
9. W przypadku niejednoznaczności wybierz najprostsze poprawne rozwiązanie.


Schemat danych wejściowych (format JSON): {schema_prompt_full}

Twoim zadaniem jest wygenerować obiekt w formacie JSON zgodny z powyższym schematem danych wejściowych na podstawie opisu użytkownika. Podaj też endpoint do jakiego to powinno być dodane.

Aktualny stan bazy danych: {state_prompt_full}
Użytkownik: {txt}

Odpowiedź w formacie JSON:
"""
            all_prompts.append(full_prompt)

            response_code = 0
            pred_ep = "NIEZNANY"
            gen_json = {}
            diff = {}
            precision = 0.0
            
            try:
                chat = model.start_chat()
                response = safe_send(chat, full_prompt)
                resp_text = response.text.strip()
                
                pred_ep_raw, clean_json_text = extract_endpoint_and_json(resp_text)
                if pred_ep_raw != "NIEZNANY":
                    pred_ep = norm_ep(pred_ep_raw)
                
                try:
                    gen_json = json.loads(clean_json_text)
                    
                    if pred_ep == "NIEZNANY":
                         if "endpoint" in gen_json:
                             pred_ep = norm_ep(gen_json["endpoint"])

                    expected_fixed = try_fix_json(expected_json_str)
                    ref_json = json.loads(expected_fixed)

                    if "data" in gen_json and isinstance(gen_json["data"], dict) and "data" not in ref_json:
                        gen_json = gen_json["data"]
                    elif "body" in gen_json and isinstance(gen_json["body"], dict) and "body" not in ref_json:
                        gen_json = gen_json["body"]
                    
                    if "endpoint" in gen_json and "endpoint" not in ref_json:
                        gen_json.pop("endpoint", None)
                    
                    precision, diff = compare_jsons(ref_json, gen_json)
                    
                    response_code = 200
                    
                except json.JSONDecodeError as je:
                    print(f"JSON Error: {je}")
                    gen_json = {"error": "JSONDecodeError", "raw": clean_json_text}
                    response_code = 0

            except Exception as e:
                print(f"Gemini Error: {e}")
                resp_text = str(e)
                response_code = 0
            
            y_true.append(true_ep)
            y_pred.append(pred_ep)
            
            samples_debug.append({
                "i": i_global,
                "text": txt,
                "true_ep": true_ep,
                "pred_ep": pred_ep,
                "ep_match": (true_ep == pred_ep),
                "precision": round(precision, 2),
                "gemini_raw": resp_text if 'resp_text' in locals() else "",
                "gemini_json": gen_json,
                "diff": diff,
                "expected": expected_json_str
            })

            with open("samples_debug_zero_shot.json", "w", encoding="utf-8") as f:
                json.dump(samples_debug, f, ensure_ascii=False, indent=2)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== WYNIKI ZERO-SHOT ===")
    print(f"Endpoint Accuracy: {acc:.4f}")
    
    with open("prompts_log_zero_shot.json", "w", encoding="utf-8") as f:
        json.dump(all_prompts, f, ensure_ascii=False, indent=2)
    print("Prompts saved to prompts_log_zero_shot.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--intents", type=str, default=DEFAULT_INTENTS_JSON)
    parser.add_argument("--testset", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.intents):
        raise FileNotFoundError(f"Missing intents: {args.intents}")
    if not os.path.exists(args.testset):
        raise FileNotFoundError(f"Missing testset: {args.testset}")

    evaluate(args.intents, args.testset)

if __name__ == "__main__":
    main()
