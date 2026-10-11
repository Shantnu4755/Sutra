"""Build the sample policy PDFs used by the evaluation and the demo library.

    python scripts/make_sample_pdfs.py          # eval/policies/*.txt -> eval/samples/*.pdf

A tiny dependency-free PDF writer: Helvetica text, "# " and "## " lines as bold headings,
automatic wrapping and pagination. Enough for pypdf to extract clean text with page numbers.
"""

import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINES_PER_PAGE = 50
CHARS_PER_LINE = 95


def _escape(line: str) -> str:
    return line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _layout(body: str) -> list[list[tuple[str, str]]]:
    """Text -> pages of (font, line)."""
    lines: list[tuple[str, str]] = []
    for raw in body.splitlines():
        if raw.startswith("# ") or raw.startswith("## "):
            if lines and lines[-1][1]:
                lines.append(("F1", ""))
            lines.append(("F2", raw.lstrip("# ").strip()))
            continue
        wrapped = textwrap.wrap(raw, CHARS_PER_LINE) or [""]
        lines.extend(("F1", part) for part in wrapped)
    return [lines[i : i + LINES_PER_PAGE] for i in range(0, len(lines), LINES_PER_PAGE)]


def text_to_pdf(title: str, body: str) -> bytes:
    pages = _layout(body)
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        4: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
        5: b"<< /Title (" + _escape(title).encode("latin-1") + b") /Producer (Sutra sample builder) >>",
    }
    kids = []
    number = 6
    for index, page in enumerate(pages, start=1):
        ops = ["BT", "14 TL", "56 790 Td"]
        for font, line in page:
            size = "10.5" if font == "F1" else "11.5"
            ops.append(f"/{font} {size} Tf ({_escape(line)}) Tj T*")
        ops += ["ET", "BT /F1 8 Tf 270 30 Td", f"(Page {index} of {len(pages)}) Tj", "ET"]
        stream = "\n".join(ops).encode("latin-1")
        objects[number + 1] = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
        objects[number] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            b"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents %d 0 R >>" % (number + 1)
        )
        kids.append(number)
        number += 2
    objects[2] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (
        b" ".join(b"%d 0 R" % k for k in kids),
        len(kids),
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for num in sorted(objects):
        offsets[num] = len(out)
        out += b"%d 0 obj\n" % num + objects[num] + b"\nendobj\n"
    xref_at = len(out)
    size = max(objects) + 1
    out += b"xref\n0 %d\n0000000000 65535 f \n" % size
    for num in range(1, size):
        out += b"%010d 00000 n \n" % offsets[num]
    out += b"trailer\n<< /Size %d /Root 1 0 R /Info 5 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (size, xref_at)
    return bytes(out)


def main() -> int:
    for source in sorted((ROOT / "eval" / "policies").glob("*.txt")):
        body = source.read_text()
        title = body.splitlines()[0].lstrip("# ").strip()
        target = ROOT / "eval" / "samples" / f"{source.stem}.pdf"
        target.write_bytes(text_to_pdf(title, body))
        sys.stdout.write(f"{target.relative_to(ROOT)}\t{title}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
