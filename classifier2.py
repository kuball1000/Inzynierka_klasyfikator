from dataclasses import dataclass
from typing import List, Dict, Tuple, Counter
import os, json, re, unicodedata, hashlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- konfiguracja ---
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
EMB_CACHE_PATH = "intent_embeddings_knn.npz"
SIM_THRESHOLD = 0.42
MAYBE_THRESHOLD = 0.32
DEFAULT_INTENTS_JSON = "intents2.json"
DEFAULT_K = 5  # Default K for KNN

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
    all_endpoints = []
    all_examples = []
    
    for ep, spec in intents.items():
        exs = spec["examples"]
        for ex in exs:
            all_endpoints.append(ep)
            all_examples.append(ex)
            
    if not all_examples:
        return IntentIndex([], np.array([]), [], intents_hash)

    embeddings = embed_passages(all_examples)
    
    return IntentIndex(all_endpoints, embeddings, all_examples, intents_hash)

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
    k_neighbors: int = DEFAULT_K
) -> Dict:
    q_emb = embed_queries([user_text])
    
    # Oblicz podobieństwo względem WSZYSTKICH przykładów
    sims = cosine_similarity(q_emb, idx.embeddings)[0]
    
    # Pobierz top K indeksów
    top_k_indices = np.argsort(sims)[-k_neighbors:][::-1]
    
    best_score_single = sims[top_k_indices[0]]
    
    # Vote
    votes = []
    for i in top_k_indices:
        ep = idx.endpoints[i]
        score = sims[i]
        votes.append((ep, score))
        
    # Zliczanie głosów: Suma wyników per endpoint dla top K
    vote_scores = {}
    for ep, score in votes:
        vote_scores[ep] = vote_scores.get(ep, 0.0) + score
        
    best_ep = max(vote_scores, key=vote_scores.get)
    
    best_ep_scores = [v[1] for v in votes if v[0] == best_ep]
    final_score = max(best_ep_scores) if best_ep_scores else 0.0
    
    # Dynamiczne progi
    # Uwaga: Dynamiczne progowanie oparte na średniej/odchyleniu w classifier1 bazowało na rozkładzie PODOBIEŃSTW DO CENTROIDÓW.
    # Tutaj mamy podobieństwa do wszystkich przykładów. Rozkład będzie inny (dużo niskich wyników).
    # Na razie pozostańmy przy stałych progach lub uproszczonej logice.
    
    if sim_threshold is None: sim_threshold = SIM_THRESHOLD
    if maybe_threshold is None: maybe_threshold = MAYBE_THRESHOLD

    if final_score >= sim_threshold:
        return {"endpoint": best_ep, "score": round(final_score, 4), "method": f"knn_semantic_k{k_neighbors}"}

    if final_score >= maybe_threshold:
        ep_kw, reason = keyword_fallback(user_text, intents)
        if ep_kw:
            return {"endpoint": ep_kw, "score": round(final_score, 4), "method": "knn_semantic+fallback", "debug": reason}
        return {"endpoint": best_ep, "score": round(final_score, 4), "method": "knn_semantic_low"}

    ep_kw, reason = keyword_fallback(user_text, intents)
    if ep_kw:
        return {"endpoint": ep_kw, "score": round(final_score, 4), "method": "fallback", "debug": reason}

    return {"endpoint": "", "score": round(final_score, 4), "method": "unknown"}


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
    p.add_argument("--k", type=int, default=DEFAULT_K) # Add K parameter
    a = p.parse_args()
    intents = load_intents_from_json(a.intents)
    idx = build_or_load_index(intents, _file_sha1(a.intents), a.rebuild)
    text = a.text or "Weterynarz Marka podała koniu Mewie 2 tabletki leku przeciwbólowego."
    print(json.dumps(predict_endpoint(text, idx, intents, k_neighbors=a.k), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    demo()
