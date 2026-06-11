from pathlib import Path
import fitz


def extract_text(pdf_path: Path, cache_dir: Path) -> str:
    """Extract text from PDF, using cached markdown if available."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{pdf_path.stem}.md"

    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    doc = fitz.open(str(pdf_path))
    lines = []

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] != 0:
                continue
            for line_info in block["lines"]:
                text = "".join(s["text"] for s in line_info["spans"]).strip()
                if text:
                    lines.append(text)
            lines.append("")

    doc.close()
    text = "\n".join(lines)
    cache_path.write_text(text, encoding="utf-8")
    return text
