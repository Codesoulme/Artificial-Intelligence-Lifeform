"""Audit metadata and cards for gated Chinese corpus candidates only."""
import hashlib
import json
from pathlib import Path
import urllib.request
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/p129-corpus-candidate-metadata-20261109"
REVISION = {"cci": None, "skypile": None}
DATASETS = {"cci": "BAAI/CCI3-HQ", "skypile": "Skywork/SkyPile-150B"}


def get(url):
    with urllib.request.urlopen(url, timeout=60) as f: return f.read()


def run():
    if OUT.exists(): raise FileExistsError(OUT)
    result = {"status": "metadata_only_not_training", "datasets": {}}
    OUT.mkdir(parents=False)
    for key, dataset in DATASETS.items():
        api = json.loads(get(f"https://huggingface.co/api/datasets/{dataset}"))
        readme_url = f"https://huggingface.co/datasets/{dataset}/raw/main/README.md"
        try:
            readme = get(readme_url)
            readme_meta = {"url": readme_url, "bytes": len(readme), "sha256": hashlib.sha256(readme).hexdigest()}
        except HTTPError as exc:
            readme_meta = {"url": readme_url, "error": f"HTTP {exc.code}"}
        result["datasets"][key] = {
            "id": dataset, "gated": api.get("gated"), "private": api.get("private"),
            "downloads": api.get("downloads"), "likes": api.get("likes"),
            "tags": api.get("tags", [])[:20], "card_data": api.get("cardData", {}),
            "readme": readme_meta,
        }
    result["limitations"] = [
        "Metadata and README statements do not prove row-level quality, rights, privacy or cross-corpus isolation.",
        "No gated access was requested and no dataset body was downloaded.",
        "No training candidate or approval was produced.",
    ]
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__": print(json.dumps(run(), ensure_ascii=False, indent=2))
