"""Convert the .docx design documents to Markdown, preserving structure.

Headings come from paragraph styles, ASCII diagrams from PreformattedText runs
(grouped into fenced blocks), and tables are rebuilt from the table XML rather
than flattened, so they render on GitHub.
"""
import re, sys, zipfile, html
from pathlib import Path
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
HEADING = {"Heading1": "##", "Heading2": "###", "Heading3": "####"}


def para_text(p):
    out = []
    for node in p.iter():
        if node.tag == f"{W}t":
            out.append(node.text or "")
        elif node.tag == f"{W}tab":
            out.append("    ")
        elif node.tag == f"{W}br":
            out.append("\n")
    return "".join(out)


def style_of(p):
    s = p.find(f"{W}pPr/{W}pStyle")
    return s.get(f"{W}val") if s is not None else "BodyText"


def is_list(p):
    return p.find(f"{W}pPr/{W}numPr") is not None


def table_md(tbl):
    rows = []
    for tr in tbl.findall(f"{W}tr"):
        cells = [" ".join(para_text(p).split()) for tc in tr.findall(f"{W}tc")
                 for p in [tc]][:0]  # placeholder
        cells = []
        for tc in tr.findall(f"{W}tc"):
            text = " ".join(" ".join(para_text(p) for p in tc.findall(f".//{W}p")).split())
            cells.append(text.replace("|", "\\|"))
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    head, *body = rows
    md = ["| " + " | ".join(head) + " |",
          "|" + "|".join(["---"] * width) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(md)


def convert(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    body = root.find(f"{W}body")
    out, pre = [], []

    def flush_pre():
        if pre:
            block = "\n".join(pre).rstrip()
            if block.strip():
                out.append("```text\n" + block + "\n```")
            pre.clear()

    for el in body:
        if el.tag == f"{W}tbl":
            flush_pre()
            t = table_md(el)
            if t:
                out.append(t)
            continue
        if el.tag != f"{W}p":
            continue
        style = style_of(el)
        text = para_text(el)
        if style == "PreformattedText":
            pre.append(text.rstrip())
            continue
        flush_pre()
        stripped = text.strip()
        if style == "HorizontalLine":
            if out and out[-1] != "---":
                out.append("---")
        elif not stripped:
            continue
        elif style in HEADING:
            # The first heading repeats the filename; the generated H1 covers it.
            if not out and stripped.lower().endswith(path.stem.lower()):
                continue
            out.append(f"{HEADING[style]} {stripped}")
        elif style == "BlockQuotation":
            out.append(f"> {stripped}")
        elif is_list(el):
            out.append(f"- {stripped}")
        else:
            out.append(stripped)
    flush_pre()

    md = "\n\n".join(out)
    # Consecutive bullets should be one tight list, not paragraph-spaced items.
    md = re.sub(r"(?m)^(- .+)\n\n(?=- )", r"\1\n", md)
    md = re.sub(r"(?m)^(- .+)\n\n(?=- )", r"\1\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md


src, dest = Path("docx"), Path("docs/design")
NAMES = {
    "Architecture Overview": "architecture-overview.md",
    "Project Requirements": "requirements.md",
    "Architecture Decision Register": "decision-register.md",
    "Architecture Spikes": "spikes.md",
    "Project Charter": "charter.md",
}
for f in sorted(src.glob("*.docx")):
    md = convert(f)
    title = f.stem
    header = (
        f"# {title}\n\n"
        f"> Converted from `docx/{f.name}`, which remains the editing source.\n"
        f"> Regenerate with `python scripts/docx2md.py` after editing the Word file.\n"
    )
    out = dest / NAMES.get(title, f.stem.lower().replace(" ", "-") + ".md")
    out.write_text(header + "\n" + md + "\n", encoding="utf-8")
    print(f"  {f.name:42} -> {out}  ({len(md.splitlines())} lines)")
