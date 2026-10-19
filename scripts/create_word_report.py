from pathlib import Path
import re

import markdown
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "report" / "REPORT.md"
OUTPUT = ROOT / "report" / "REPORT.docx"

text = SOURCE.read_text(encoding="utf-8")

# Remove YAML front matter; its metadata is added to the document below.
if text.startswith("---"):
    parts = text.split("---", 2)
    if len(parts) == 3:
        text = parts[2].strip()

# Replace Markdown images with temporary tokens, preserving only real images.
images = {}

def image_replacement(match):
    alt = match.group(1).strip()
    raw_path = match.group(2).strip()
    raw_path = re.sub(r"\s*\{[^}]*\}\s*$", "", raw_path)
    path = (SOURCE.parent / raw_path).resolve()

    if not path.is_file():
        return f"\n\n[Image omitted: {alt} — file not present in repository]\n\n"

    token = f"IMAGE_TOKEN_{len(images)}"
    images[token] = (path, alt)
    return f"\n\n{token}\n\n"

text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", image_replacement, text)

# Convert remaining Markdown to HTML for parsing headings, paragraphs, lists and tables.
html = markdown.markdown(
    text,
    extensions=["tables", "fenced_code", "sane_lists"],
)

from html.parser import HTMLParser
from html import unescape

class BlocksParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.blocks = []
        self.current = None
        self.stack = []
        self.in_table = False
        self.table = []
        self.row = None
        self.cell = None
        self.in_pre = False

    def handle_starttag(self, tag, attrs):
        if tag in ("h1", "h2", "h3", "h4", "p", "li", "pre"):
            self.current = {"tag": tag, "text": ""}
            self.in_pre = tag == "pre"
        elif tag == "table":
            self.in_table = True
            self.table = []
        elif self.in_table and tag == "tr":
            self.row = []
        elif self.in_table and tag in ("th", "td"):
            self.cell = {"text": "", "header": tag == "th"}
        elif tag in ("strong", "b", "em", "code"):
            self.stack.append(tag)
        elif tag == "br" and self.current is not None:
            self.current["text"] += "\n"

    def handle_endtag(self, tag):
        if self.in_table:
            if tag in ("th", "td") and self.cell is not None:
                self.row.append(self.cell)
                self.cell = None
            elif tag == "tr" and self.row is not None:
                if any(c["text"].strip() for c in self.row):
                    self.table.append(self.row)
                self.row = None
            elif tag == "table":
                self.blocks.append({"tag": "table", "rows": self.table})
                self.in_table = False
            return

        if tag in ("h1", "h2", "h3", "h4", "p", "li", "pre"):
            if self.current is not None:
                self.blocks.append(self.current)
            self.current = None
            self.in_pre = False
        elif tag in ("strong", "b", "em", "code") and self.stack:
            self.stack.pop()

    def handle_data(self, data):
        if self.in_table and self.cell is not None:
            self.cell["text"] += data
        elif self.current is not None:
            self.current["text"] += data
        elif data.strip():
            self.blocks.append({"tag": "p", "text": data})

parser = BlocksParser()
parser.feed(html)

doc = Document()
section = doc.sections[0]
section.top_margin = Inches(0.7)
section.bottom_margin = Inches(0.7)
section.left_margin = Inches(0.75)
section.right_margin = Inches(0.75)

styles = doc.styles
styles["Normal"].font.name = "Calibri"
styles["Normal"].font.size = Pt(10)
styles["Normal"].paragraph_format.space_after = Pt(5)
for style_name, size in [("Title", 22), ("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 11)]:
    styles[style_name].font.name = "Calibri"
    styles[style_name].font.size = Pt(size)
    styles[style_name].font.color.rgb = RGBColor(31, 78, 121)

title = doc.add_paragraph()
title.style = "Title"
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.add_run("MLOps Assignment 1\nHeart Disease Risk Prediction")

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("Machine Learning Operations (AIMLCZG523), BITS Pilani")
r.italic = True

for line in [
    "Author: Sudarsan S",
    "BITS ID: 2025AE05524",
    "Group/Section: N/A",
    "Submission date: 20 October 2026",
    "Repository: https://github.com/Sudarsas2410/ml_ops",
]:
    p = doc.add_paragraph(line)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_page_break()

def add_inline_text(paragraph, value):
    # Basic inline formatting for bold, italic and code; retain readable text.
    pattern = re.compile(r"(\*\*.*?\*\*|\*.*?\*|`.*?`)")
    for part in pattern.split(value):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            paragraph.add_run(part[2:-2]).bold = True
        elif part.startswith("*") and part.endswith("*"):
            paragraph.add_run(part[1:-1]).italic = True
        elif part.startswith("`") and part.endswith("`"):
            run = paragraph.add_run(part[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9)
        else:
            paragraph.add_run(part)

for block in parser.blocks:
    tag = block["tag"]
    value = block.get("text", "").strip()

    if tag == "table":
        rows = block["rows"]
        if not rows:
            continue
        col_count = max(len(row) for row in rows)
        table = doc.add_table(rows=0, cols=col_count)
        table.style = "Light Shading Accent 1"
        for row_data in rows:
            cells = table.add_row().cells
            for i in range(col_count):
                cell_text = row_data[i]["text"].strip() if i < len(row_data) else ""
                cells[i].text = cell_text
                if i < len(row_data) and row_data[i]["header"]:
                    for run in cells[i].paragraphs[0].runs:
                        run.bold = True
        doc.add_paragraph()
        continue

    if not value:
        continue

    token_match = re.fullmatch(r"IMAGE_TOKEN_\d+", value)
    if token_match:
        path, alt = images[value]
        try:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run().add_picture(str(path), width=Inches(5.8))
            caption = doc.add_paragraph(alt)
            caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if caption.runs:
                caption.runs[0].italic = True
        except Exception as exc:
            doc.add_paragraph(f"[Image could not be embedded: {alt} ({exc})]")
        continue

    if value.startswith("[Image omitted:"):
        p = doc.add_paragraph(value)
        p.runs[0].italic = True
        continue

    if tag == "h1":
        doc.add_heading(value, level=1)
    elif tag == "h2":
        doc.add_heading(value, level=2)
    elif tag == "h3":
        doc.add_heading(value, level=3)
    elif tag == "h4":
        doc.add_heading(value, level=4)
    elif tag == "pre":
        p = doc.add_paragraph()
        run = p.add_run(value)
        run.font.name = "Consolas"
        run.font.size = Pt(8)
    elif tag == "li":
        p = doc.add_paragraph(style="List Bullet")
        add_inline_text(p, value)
    else:
        p = doc.add_paragraph()
        add_inline_text(p, value)

# Add a footer with page numbers as plain text label.
for sec in doc.sections:
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("MLOps Assignment 1 | Sudarsan S | 2025AE05524")

doc.save(OUTPUT)
print(f"Created: {OUTPUT}")
print(f"Embedded existing images: {len(images)}")
print("Missing image references were marked as omitted, not fabricated.")
