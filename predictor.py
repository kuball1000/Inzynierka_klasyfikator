# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
from classifier import (
    build_or_load_index,
    predict_endpoint,
    load_intents_from_json,
    _file_sha1,
    DEFAULT_INTENTS_JSON,
)

load_dotenv()

INDEX = None
INTENTS = None
INTENTS_HASH = None
INTENTS_PATH = os.getenv("INTENTS_PATH", DEFAULT_INTENTS_JSON)


def norm_ep(ep: str) -> str:
    """Ujednolicenie ścieżek endpointów (np. '/api/konie' -> 'api/konie')."""
    ep = (ep or "").strip()
    if ep.startswith("/"):
        ep = ep[1:]
    return ep


def load_index(force_rebuild: bool = False):
    """Ładuje lub buduje indeks embeddingów na podstawie pliku intents JSON."""
    global INDEX, INTENTS, INTENTS_HASH

    if not os.path.exists(INTENTS_PATH):
        raise FileNotFoundError(f"Nie znaleziono pliku intents: {INTENTS_PATH}")

    INTENTS = load_intents_from_json(INTENTS_PATH)
    INTENTS_HASH = _file_sha1(INTENTS_PATH)
    INDEX = build_or_load_index(
        intents=INTENTS,
        intents_hash=INTENTS_HASH,
        force_rebuild=force_rebuild,
    )


def predict_from_text(txt: str) -> str:
    """Zwraca przewidziany endpoint na podstawie promptu użytkownika."""
    global INDEX, INTENTS

    if INDEX is None or INTENTS is None:
        load_index(force_rebuild=False)

    assert INDEX is not None
    assert INTENTS is not None
    out = predict_endpoint(txt, INDEX, intents=INTENTS)
    return norm_ep(out.get("endpoint", ""))
