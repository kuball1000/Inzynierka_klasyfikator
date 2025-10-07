# -*- coding: utf-8 -*-
"""
Lokalny klasyfikator intencji -> endpointów API (PL)
- bez zewnętrznych API
- embeddings: multilingual (polski)
- dopasowanie: cosinus + próg + fallback słów-kluczy
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

# 1) WYBIERZ MODEL (oba działają offline po jednorazowym pobraniu):
#    'intfloat/multilingual-e5-base'  - dokładniejszy, większy
#    'intfloat/multilingual-e5-small' - szybszy, mniejszy
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

# 2) KONFIG: progi i ścieżki
EMB_CACHE_PATH = "intent_embeddings.npz"
SIM_THRESHOLD = 0.42   # minimalna sensowna podobność (dopasowanie "na pewno")
MAYBE_THRESHOLD = 0.32 # strefa "może" – jeśli < SIM_THRESHOLD, ale >= MAYBE_THRESHOLD i brak lepszego wyniku, użyjemy fallbacku/heurystyk

# 3) INTENCJE: słownik endpoint -> przykładowe frazy i słowa klucze
#    Dodawaj/zmieniaj przykłady – to "uczenie" few-shot bez trenowania.
DEFAULT_INTENTS_JSON = "intents.json"

# --- MODELE / EMBEDDINGS ---

# Ładujemy sentence-transformers dopiero na żądanie, by plik startował szybko
_model = None
def get_model():
    global _model
    if _model is None:
        # sentence-transformers opiera się na HF Transformers, ale działa w 100% lokalnie
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model

def e5_encode(texts: List[str]) -> np.ndarray:
    """
    Modele E5 oczekują prefiksów 'query:' dla zapytań i 'passage:' dla dokumentów.
    """
    model = get_model()
    return model.encode(texts, normalize_embeddings=True)

def embed_queries(texts: List[str]) -> np.ndarray:
    return e5_encode([f"query: {t}" for t in texts])

def embed_passages(texts: List[str]) -> np.ndarray:
    return e5_encode([f"passage: {t}" for t in texts])

# --- INTENTS: ładowanie + walidacja ---

def _file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def load_intents_from_json(path: str) -> Dict[str, Dict[str, List[str]]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nie znaleziono pliku z intencjami: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Prosta walidacja struktury
    if not isinstance(data, dict):
        raise ValueError("Plik intents JSON musi być słownikiem endpoint -> {examples, keywords}.")

    cleaned: Dict[str, Dict[str, List[str]]] = {}
    for endpoint, spec in data.items():
        if not isinstance(spec, dict):
            continue
        exs = spec.get("examples", [])
        kws = spec.get("keywords", [])
        if not isinstance(exs, list) or not all(isinstance(x, str) for x in exs):
            exs = []
        if not isinstance(kws, list) or not all(isinstance(x, str) for x in kws):
            kws = []
        if exs:  # bierzemy tylko endpointy z co najmniej jednym przykładem
            cleaned[endpoint] = {"examples": exs, "keywords": kws}
    if not cleaned:
        raise ValueError("Brak poprawnych intencji (co najmniej jeden endpoint musi mieć examples[]).")
    return cleaned


# --- BAZA PRZYKŁADÓW -> WEKTORY ---

@dataclass
class IntentIndex:
    endpoints: List[str]
    texts: List[str]
    labels: List[int]
    embeddings: np.ndarray  # shape: (N, D)
    intents_hash: str

def build_intent_index(intents: Dict[str, Dict[str, List[str]]], intents_hash: str) -> IntentIndex:
    endpoints, texts, labels = [], [], []
    for idx, (endpoint, spec) in enumerate(intents.items()):
        exs = spec.get("examples", [])
        if not exs:
            continue
        for t in exs:
            endpoints.append(endpoint)
            texts.append(t)
            labels.append(idx)
    if not texts:
        raise ValueError("Brak przykładów do zbudowania indeksu.")
    emb = embed_passages(texts)
    return IntentIndex(
        endpoints=endpoints,
        texts=texts,
        labels=labels,
        embeddings=emb,
        intents_hash=intents_hash
    )

def save_index(idx: IntentIndex, path: str = EMB_CACHE_PATH):
    np.savez_compressed(
        path,
        endpoints=np.array(idx.endpoints, dtype=object),
        texts=np.array(idx.texts, dtype=object),
        labels=np.array(idx.labels, dtype=np.int32),
        embeddings=idx.embeddings.astype(np.float32),
        intents_hash=np.array(idx.intents_hash, dtype=object),
    )

def load_index(path: str = EMB_CACHE_PATH) -> IntentIndex:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nie znaleziono indeksu: {path}")
    data = np.load(path, allow_pickle=True)
    return IntentIndex(
        endpoints=data["endpoints"].tolist(),
        texts=data["texts"].tolist(),
        labels=data["labels"].tolist(),
        embeddings=data["embeddings"],
        intents_hash=str(data["intents_hash"]),
    )

# --- KLASYFIKACJA ---

def keyword_fallback(text: str, intents: Dict[str, Dict[str, List[str]]]) -> Tuple[str, str]:
    """
    Bardzo prosty fallback: jeśli są bezdyskusyjne słowa-klucze, wybierz ten endpoint.
    Zwraca (endpoint, powód)
    """
    t = text.lower()
    hits = []
    for endpoint, spec in intents.items():
        for kw in spec.get("keywords", []):
            if kw.lower() in t:
                hits.append((endpoint, kw))
                break
    if not hits:
        return "", ""
    # jeżeli wiele – wybierz z najdłuższym słowem (często bardziej specyficzne)
    hits.sort(key=lambda x: len(x[1]), reverse=True)
    ep, kw = hits[0]
    return ep, f"fallback słowo-klucz: '{kw}'"

def predict_endpoint(
    user_text: str,
    idx: IntentIndex,
    intents: Dict[str, Dict[str, List[str]]],
    sim_threshold: float = SIM_THRESHOLD,
    maybe_threshold: float = MAYBE_THRESHOLD
) -> Dict:
    """
    Zwraca dict z polami:
    - endpoint
    - score (cosine)
    - method ('semantic' | 'semantic+fallback' | 'fallback' | 'unknown')
    - debug (kontekst dopasowania)
    """
    q_emb = embed_queries([user_text])
    sims = cosine_similarity(q_emb, idx.embeddings)[0]  # (N,)
    best_i = int(np.argmax(sims))
    best_score = float(sims[best_i])
    best_endpoint = idx.endpoints[best_i]

    # przypadek oczywisty
    if best_score >= sim_threshold:
        return {
            "endpoint": best_endpoint,
            "score": round(best_score, 4),
            "method": "semantic",
            "debug": {
                "matched_example": idx.texts[best_i]
            }
        }

    # strefa "może": spróbuj wspomóc się słowami-kluczami
    if best_score >= maybe_threshold:
        ep_kw, reason = keyword_fallback(user_text, intents)
        if ep_kw and ep_kw != best_endpoint:
            # jeżeli fallback wskazuje inny, ale sensowny endpoint, a różnica mała – podbij go
            return {
                "endpoint": ep_kw,
                "score": round(best_score, 4),
                "method": "semantic+fallback",
                "debug": {
                    "semantic_candidate": best_endpoint,
                    "matched_example": idx.texts[best_i],
                    "fallback_reason": reason
                }
            }
        else:
            return {
                "endpoint": best_endpoint,
                "score": round(best_score, 4),
                "method": "semantic",
                "debug": {
                    "matched_example": idx.texts[best_i],
                    "note": "wynik poniżej progu pewności, ale brak lepszego dopasowania"
                }
            }

    # słaby wynik – użyj czystego fallbacku
    ep_kw, reason = keyword_fallback(user_text, intents)
    if ep_kw:
        return {
            "endpoint": ep_kw,
            "score": round(best_score, 4),
            "method": "fallback",
            "debug": {"fallback_reason": reason, "semantic_top": best_endpoint, "semantic_score": round(best_score, 4)}
        }

    # nic nie pasuje
    return {
        "endpoint": "",
        "score": round(best_score, 4),
        "method": "unknown",
        "debug": {"semantic_top": best_endpoint, "semantic_score": round(best_score, 4)}
    }

# --- POMOCNICZE: budowa/odświeżenie indeksu ---

def build_or_load_index(intents: Dict[str, Dict[str, List[str]]], intents_hash: str, force_rebuild: bool = False) -> IntentIndex:
    """
    Ładuje cache, jeśli istnieje i hash pliku intents jest zgodny; w przeciwnym razie buduje od nowa.
    """
    if (not force_rebuild) and os.path.exists(EMB_CACHE_PATH):
        try:
            idx = load_index(EMB_CACHE_PATH)
            if idx.intents_hash == intents_hash:
                return idx
        except Exception:
            pass  # jeśli coś nie gra, po prostu przebuduj
    idx = build_intent_index(intents, intents_hash)
    save_index(idx, EMB_CACHE_PATH)
    return idx

# --- DEMO CLI ---

def demo():
    import argparse
    parser = argparse.ArgumentParser(description="Lokalny klasyfikator intencji (PL) -> endpoint API")
    parser.add_argument("--rebuild", action="store_true", help="Przebuduj indeks embeddingów od zera")
    parser.add_argument("--text", type=str, help="Tekst wejściowy w języku naturalnym")
    parser.add_argument("--intents", type=str, default=DEFAULT_INTENTS_JSON, help="Ścieżka do pliku intents JSON")
    args = parser.parse_args()

    intents_path = args.intents
    intents = load_intents_from_json(intents_path)
    intents_hash = _file_sha1(intents_path)
    idx = build_or_load_index(intents=intents, intents_hash=intents_hash, force_rebuild=args.rebuild)

    text = args.text or "Dodaj Mi Klacz Justynę która wczoraj przyjechała do stajni, będzie koniem sportowym"
    out = predict_endpoint(text, idx, intents=intents)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    demo()