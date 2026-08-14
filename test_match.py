"""
Script de test manuel pour /api/match, qui évite le problème d'échappement
JSON des textes multi-lignes copiés-collés directement dans curl ou /docs.

Usage :
    1. Mettez le texte du CV dans cv.txt (sauts de ligne normaux, pas d'échappement)
    2. Mettez le texte de l'offre dans job.txt
    3. python3 test_match.py
"""

import json
import urllib.request

with open("cv.txt", encoding="utf-8") as f:
    cv_text = f.read()

with open("job.txt", encoding="utf-8") as f:
    job_text = f.read()

payload = json.dumps({"cv_text": cv_text, "job_text": job_text}).encode("utf-8")

req = urllib.request.Request(
    "http://localhost:8000/api/match",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(req) as response:
    result = json.loads(response.read())

print(json.dumps(result, indent=2, ensure_ascii=False))