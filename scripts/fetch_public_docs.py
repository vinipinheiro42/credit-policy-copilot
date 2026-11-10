"""Download the public regulatory documents used in the corpus.

Run: python scripts/fetch_public_docs.py
Then: python scripts/manifest_hashes.py --write   (records sha256 in MANIFEST.yaml)

Sources and reuse basis are documented in data/corpus/MANIFEST.yaml.
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "corpus" / "public"

PDFS = {
    "occ-ch-credit-card-lending.pdf": "https://www.occ.gov/publications-and-resources/publications/comptrollers-handbook/files/credit-card-lending/pub-ch-credit-card.pdf",
    "occ-ch-retail-lending.pdf": "https://www.occ.treas.gov/publications-and-resources/publications/comptrollers-handbook/files/retail-lending/pub-ch-retail-lending.pdf",
}
BCB_API = "https://www.bcb.gov.br/api/conteudo/app/normativos/exibenormativo"
BCB_ACTS = {"cmn-resolucao-4966.md": ("Resolução CMN", "4966")}


def fetch_pdf(client: httpx.Client, name: str, url: str) -> None:
    r = client.get(url)
    r.raise_for_status()
    if not r.content.startswith(b"%PDF"):
        raise RuntimeError(f"{url} did not return a PDF (content-type {r.headers.get('content-type')})")
    (OUT / name).write_bytes(r.content)
    print(f"saved {name} ({len(r.content):,} bytes)")


def bcb_html_to_markdown(payload: dict, source_url: str) -> str:
    c = payload["conteudo"][0]
    body = c["Texto"]
    body = re.sub(r"<s[\s>].*?</s>", "", body, flags=re.DOTALL | re.IGNORECASE)  # superseded wording
    paras = re.findall(r"<p[^>]*>(.*?)</p>", body, flags=re.DOTALL | re.IGNORECASE)
    lines = []
    for p in paras:
        p = re.sub(r"<br\s*/?>", "\n", p, flags=re.IGNORECASE)
        for part in html.unescape(re.sub(r"<[^>]+>", "", p)).split("\n"):
            t = re.sub(r"\s+", " ", part.replace("\u200b", "")).strip()
            if t:
                lines.append(t)

    out: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if re.match(r"^CAP[ÍI]TULO [IVXLC]+$", ln):
            out += ["", f"## {ln} — {nxt}", ""]
            i += 2
            continue
        if re.match(r"^Seção [IVXLC]+$", ln):
            out += ["", f"### {ln} — {nxt}", ""]
            i += 2
            continue
        if re.match(r"^Subseção [IVXLC]+$", ln):
            out += ["", f"#### {ln} — {nxt}", ""]
            i += 2
            continue
        m = re.match(r"^(Art\. \d+(?:\.\d+)?(?:º|°|-[A-Z])?)\s*(.*)$", ln)
        if m:
            out += ["", f"##### {m.group(1)}", "", m.group(2)]
        else:
            out += ["", ln]
        i += 1

    front = [
        "---",
        f'title: "{c["Titulo"]}"',
        "language: pt",
        "synthetic: false",
        f'source_url: "{source_url}"',
        f'published: "{c.get("DOU", "")}"',
        f'version: "{c.get("VersaoNormativo")}"',
        "---",
        "",
        "> Unofficial plain-text rendering of the consolidated text published by Banco Central do Brasil.",
        "> Superseded (struck-through) wording was removed. The official text prevails.",
        "",
        f"# {c['Titulo']}",
        "",
        c.get("Assunto", ""),
    ]
    return "\n".join(front + out).strip() + "\n"


def fetch_bcb(client: httpx.Client, name: str, tipo: str, numero: str) -> None:
    r = client.get(BCB_API, params={"p1": tipo, "p2": numero})
    r.raise_for_status()
    human_url = f"https://www.bcb.gov.br/estabilidadefinanceira/exibenormativo?tipo={tipo}&numero={numero}"
    md = bcb_html_to_markdown(r.json(), human_url)
    (OUT / name).write_text(md, encoding="utf-8")
    raw = OUT / "raw"
    raw.mkdir(exist_ok=True)
    (raw / name.replace(".md", ".json")).write_text(
        json.dumps(r.json(), ensure_ascii=False), encoding="utf-8"
    )
    print(f"saved {name} ({len(md):,} chars)")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    failures = 0
    with httpx.Client(
        timeout=60, follow_redirects=True, headers={"User-Agent": "credit-policy-copilot/0.1"}
    ) as cl:
        for name, url in PDFS.items():
            try:
                fetch_pdf(cl, name, url)
            except Exception as exc:
                failures += 1
                print(f"FAILED {name}: {exc}", file=sys.stderr)
        for name, (tipo, numero) in BCB_ACTS.items():
            try:
                fetch_bcb(cl, name, tipo, numero)
            except Exception as exc:
                failures += 1
                print(f"FAILED {name}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
