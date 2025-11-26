from dataclasses import dataclass
from typing import List, Dict, Tuple
import os, json, re, unicodedata, hashlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- konfiguracja ---
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
EMB_CACHE_PATH = "intent_embeddings.npz"
SIM_THRESHOLD = 0.42
MAYBE_THRESHOLD = 0.32
DEFAULT_INTENTS_JSON = "intents.json"

_model = None
def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model

def normalize_text(t: str) -> str:
    t = t.lower().strip()
    t = unicodedata.normalize("NFKD", t)
    t = re.sub(r"[^a-ząćęłńóśźż0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t

def e5_encode(texts: List[str]) -> np.ndarray:
    model = get_model()
    return model.encode(texts, normalize_embeddings=True)

def embed_queries(texts: List[str]) -> np.ndarray:
    return e5_encode([f"query: {normalize_text(t)}" for t in texts])

def embed_passages(texts: List[str]) -> np.ndarray:
    return e5_encode([f"passage: {normalize_text(t)}" for t in texts])

def _file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def load_intents_from_json(path: str) -> Dict[str, Dict[str, List[str]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cleaned = {}
    for ep, spec in data.items():
        if isinstance(spec, dict):
            exs = [x for x in spec.get("examples", []) if isinstance(x, str)]
            kws = [x for x in spec.get("keywords", []) if isinstance(x, str)]
            if exs:
                cleaned[ep] = {"examples": exs, "keywords": kws}
    if not cleaned:
        raise ValueError("Brak poprawnych intencji w pliku.")
    return cleaned


@dataclass
class IntentIndex:
    endpoints: List[str]
    embeddings: np.ndarray
    texts: List[str]
    intents_hash: str

def build_intent_index(intents: Dict[str, Dict[str, List[str]]], intents_hash: str) -> IntentIndex:
    endpoints, endpoint_embs, texts = [], [], []
    for ep, spec in intents.items():
        exs = spec["examples"]
        emb = embed_passages(exs)
        avg = np.mean(emb, axis=0)
        endpoints.append(ep)
        endpoint_embs.append(avg)
        texts.append("; ".join(exs[:3]))
    return IntentIndex(endpoints, np.stack(endpoint_embs, axis=0), texts, intents_hash)

def save_index(idx: IntentIndex, path: str = EMB_CACHE_PATH):
    np.savez_compressed(
        path,
        endpoints=np.array(idx.endpoints, dtype=object),
        texts=np.array(idx.texts, dtype=object),
        embeddings=idx.embeddings.astype(np.float32),
        intents_hash=np.array(idx.intents_hash, dtype=object)
    )

def load_index(path: str = EMB_CACHE_PATH) -> IntentIndex:
    data = np.load(path, allow_pickle=True)
    return IntentIndex(
        endpoints=data["endpoints"].tolist(),
        texts=data["texts"].tolist(),
        embeddings=data["embeddings"],
        intents_hash=str(data["intents_hash"])
    )


def keyword_fallback(text: str, intents: Dict[str, Dict[str, List[str]]]) -> Tuple[str, str]:
    """Prosty, stabilny fallback oparty o liczbę słów-kluczy."""
    t = text.lower()
    best_ep, best_hits = "", 0
    for ep, spec in intents.items():
        hits = sum(1 for kw in spec.get("keywords", []) if kw.lower() in t)
        if hits > best_hits:
            best_ep, best_hits = ep, hits
    return (best_ep, f"{best_hits} keyword hits") if best_hits else ("", "")

def predict_endpoint(
    user_text: str,
    idx: IntentIndex,
    intents: Dict[str, Dict[str, List[str]]],
    sim_threshold: float = None,
    maybe_threshold: float = None,
    top_k: int = 5
) -> Dict:
    q_emb = embed_queries([user_text])
    sims = cosine_similarity(q_emb, idx.embeddings)[0]

    mean_s, std_s = float(np.mean(sims)), float(np.std(sims))
    if sim_threshold is None:
        sim_threshold = max(SIM_THRESHOLD, mean_s - 0.3 * std_s)
    if maybe_threshold is None:
        maybe_threshold = max(MAYBE_THRESHOLD, mean_s - 0.6 * std_s)

    # reranking top-k (średnia, bez wag)
    top_indices = np.argsort(sims)[-top_k:]
    ep_scores = {}
    for i in top_indices:
        ep = idx.endpoints[i]
        ep_scores.setdefault(ep, []).append(sims[i])
    best_ep = max(ep_scores, key=lambda e: np.mean(ep_scores[e]))
    best_score = float(np.mean(ep_scores[best_ep]))

    if best_score >= sim_threshold:
        return {"endpoint": best_ep, "score": round(best_score, 4), "method": "semantic"}

    if best_score >= maybe_threshold:
        ep_kw, reason = keyword_fallback(user_text, intents)
        if ep_kw:
            return {"endpoint": ep_kw, "score": round(best_score, 4), "method": "semantic+fallback", "debug": reason}
        return {"endpoint": best_ep, "score": round(best_score, 4), "method": "semantic_low"}

    ep_kw, reason = keyword_fallback(user_text, intents)
    if ep_kw:
        return {"endpoint": ep_kw, "score": round(best_score, 4), "method": "fallback", "debug": reason}

    return {"endpoint": "", "score": round(best_score, 4), "method": "unknown"}


def build_or_load_index(intents: Dict[str, Dict[str, List[str]]], intents_hash: str, force_rebuild: bool = False) -> IntentIndex:
    if (not force_rebuild) and os.path.exists(EMB_CACHE_PATH):
        try:
            idx = load_index(EMB_CACHE_PATH)
            if idx.intents_hash == intents_hash:
                return idx
        except Exception:
            pass
    idx = build_intent_index(intents, intents_hash)
    save_index(idx, EMB_CACHE_PATH)
    return idx


def demo():
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("--intents", type=str, default=DEFAULT_INTENTS_JSON)
    p.add_argument("--text", type=str)
    p.add_argument("--rebuild", action="store_true")
    a = p.parse_args()
    intents = load_intents_from_json(a.intents)
    idx = build_or_load_index(intents, _file_sha1(a.intents), a.rebuild)
    text = a.text or "Dodaj konia Justynę jako konia sportowego"
    print(json.dumps(predict_endpoint(text, idx, intents), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    demo()
