# Inzynierka_klasyfikator

Lokalny klasyfikator intencji -> endpointów API (PL)
- bez zewnętrznych API
- embeddings: multilingual (polski)
- dopasowanie: cosinus + próg + fallback słów-kluczy


Ewaluacja klasyfikatora intencji -> endpointów API na zbiorze testowym (JSON),
zgodna z classifier.py, który ładuje intents z pliku JSON i cache'uje embeddings
po hash'u pliku intents.

Użycie:
```bash
python eval_classifier.py --intents intents.json --testset testset.json [--rebuild]
```

## Quickstart

```bash
uv venv
source .venv/bin/activate # bash, linux
.venv/bin/activate.ps1 # powershell, windows
uv sync
python eval_classifier.py --intents intents.json --testset testset.json
```

## Server mode

Run the server: 
```
uvicorn main:app --reload
```

Access the documentation at: <http://127.0.0.1:8000/docs>