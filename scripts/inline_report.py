#!/usr/bin/env python3
"""Write docs/report_inline.tex: report.tex with every \\input{...} expanded in place.

One self-contained source file for Overleaf or submission. The two figure PDFs in
docs/figures/ still travel with it. Generated; edit report.tex and the tables, not this.

    python scripts/inline_report.py
"""

import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"


def inline(text: str) -> str:
    def expand(match: re.Match) -> str:
        path = DOCS / match.group(1)
        return path.read_text().rstrip("\n")

    return re.sub(r"\\input\{([^}]+)\}", expand, text)


if __name__ == "__main__":
    source = inline((DOCS / "report.tex").read_text())
    assert "\\input{" not in source
    (DOCS / "report_inline.tex").write_text(source)
    print("wrote", DOCS / "report_inline.tex")
