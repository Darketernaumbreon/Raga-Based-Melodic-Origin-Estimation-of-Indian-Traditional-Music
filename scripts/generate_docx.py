"""
generate_docx.py

Converts paper/draft_v2.md into a beautifully formatted Microsoft Word (.docx) document:
  - Custom typography (Calibri/Calibri Light or Times New Roman)
  - Styled tables with dark headers, alternating rows, and clean cell padding
  - Styled callout blocks for diagrams and key equations
  - Proper heading structure (Title, Subtitle, H1, H2, H3)
  - Numbered references section
"""

import sys
import re
from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MD_PATH = PROJECT_ROOT / "paper" / "draft_v2.md"
DOCX_PATH = PROJECT_ROOT / "paper" / "Raga_Identification_Research_Paper.docx"

def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def set_table_borders(table, color="D3D3D3"):
    tblPr = table._element.xpath('w:tblPr')
    if tblPr:
        borders = parse_xml(
            f'<w:tblBorders {nsdecls("w")}>'
            f'<w:top w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
            f'<w:bottom w:val="single" w:sz="8" w:space="0" w:color="4A6572"/>'
            f'<w:left w:val="none"/>'
            f'<w:right w:val="none"/>'
            f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
            f'<w:insideV w:val="none"/>'
            f'</w:tblBorders>'
        )
        tblPr[0].append(borders)

def add_styled_paragraph(doc, text, style='Normal', space_before=0, space_after=6, line_spacing=1.15):
    p = doc.add_paragraph(style=style)
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing
    
    # Parse inline formatting (**bold**, *italic*, `code`, math)
    tokens = re.split(r'(\*\*.*?\*\*|\*.*?\*|`.*?`|\$.*?\$)', text)
    for token in tokens:
        if not token:
            continue
        if token.startswith('**') and token.endswith('**'):
            run = p.add_run(token[2:-2])
            run.bold = True
        elif token.startswith('*') and token.endswith('*'):
            run = p.add_run(token[1:-1])
            run.italic = True
        elif token.startswith('`') and token.endswith('`'):
            run = p.add_run(token[1:-1])
            run.font.name = 'Consolas'
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        elif token.startswith('$') and token.endswith('$'):
            run = p.add_run(token[1:-1])
            run.italic = True
            run.font.color.rgb = RGBColor(0x11, 0x4B, 0x7E)
        else:
            p.add_run(token)
    return p

def create_docx_from_markdown():
    print(f"Reading Markdown from: {MD_PATH}")
    md_content = MD_PATH.read_text(encoding="utf-8")
    
    doc = Document()
    
    # Page Margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Styles Setup
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    
    lines = md_content.split('\n')
    i = 0
    in_code_block = False
    code_block_lines = []
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Code block handling
        if stripped.startswith('```'):
            if in_code_block:
                # End code block -> Add styled box
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(6)
                p.paragraph_format.space_after = Pt(8)
                table = doc.add_table(rows=1, cols=1)
                table.alignment = WD_TABLE_ALIGNMENT.CENTER
                cell = table.rows[0].cells[0]
                cell.width = Inches(6.5)
                set_cell_background(cell, "F4F6F8")
                set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
                
                cp = cell.paragraphs[0]
                cp.paragraph_format.space_before = Pt(2)
                cp.paragraph_format.space_after = Pt(2)
                cp.paragraph_format.line_spacing = 1.05
                run = cp.add_run("\n".join(code_block_lines))
                run.font.name = 'Consolas'
                run.font.size = Pt(9.0)
                run.font.color.rgb = RGBColor(0x2B, 0x3E, 0x50)
                
                code_block_lines = []
                in_code_block = False
            else:
                in_code_block = True
                code_block_lines = []
            i += 1
            continue
            
        if in_code_block:
            code_block_lines.append(line)
            i += 1
            continue
            
        if not stripped:
            i += 1
            continue
            
        # Horizontal Rule
        if stripped == '---':
            i += 1
            continue
            
        # Headings
        if stripped.startswith('# '):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(18)
            p.paragraph_format.space_after = Pt(6)
            run = p.add_run(stripped[2:])
            run.font.name = 'Calibri Light'
            run.font.size = Pt(22)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D) # Navy
            i += 1
            continue
            
        if stripped.startswith('## '):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(16)
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run(stripped[3:])
            run.font.name = 'Calibri'
            run.font.size = Pt(15)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x2B, 0x4C, 0x7E)
            i += 1
            continue
            
        if stripped.startswith('### '):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(3)
            run = p.add_run(stripped[4:])
            run.font.name = 'Calibri'
            run.font.size = Pt(12.5)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x3B, 0x59, 0x98)
            i += 1
            continue

        if stripped.startswith('#### '):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run(stripped[5:])
            run.font.name = 'Calibri'
            run.font.size = Pt(11.5)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x4A, 0x55, 0x68)
            i += 1
            continue

        # Markdown Table parsing
        if stripped.startswith('|') and '|' in stripped[1:]:
            # Collect table lines
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i].strip())
                i += 1
                
            # Filter out separator line (|---|---|)
            table_lines = [tl for tl in table_lines if not re.match(r'^\|[\s\:\-]*\|', tl)]
            
            if table_lines:
                rows_data = []
                for tl in table_lines:
                    cells = [c.strip() for c in tl.split('|')[1:-1]]
                    rows_data.append(cells)
                    
                if rows_data:
                    num_cols = max(len(r) for r in rows_data)
                    table = doc.add_table(rows=len(rows_data), cols=num_cols)
                    table.alignment = WD_TABLE_ALIGNMENT.CENTER
                    set_table_borders(table)
                    
                    for r_idx, row in enumerate(rows_data):
                        tr = table.rows[r_idx]
                        is_header = (r_idx == 0)
                        
                        for c_idx, cell_text in enumerate(row):
                            if c_idx < len(tr.cells):
                                cell = tr.cells[c_idx]
                                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                                set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
                                
                                if is_header:
                                    set_cell_background(cell, "1A365D") # Dark navy header
                                else:
                                    if r_idx % 2 == 1:
                                        set_cell_background(cell, "FFFFFF")
                                    else:
                                        set_cell_background(cell, "F7FAFC") # Light tint
                                        
                                p = cell.paragraphs[0]
                                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                                p.paragraph_format.space_before = Pt(2)
                                p.paragraph_format.space_after = Pt(2)
                                p.paragraph_format.line_spacing = 1.05
                                
                                # Inline parse
                                tokens = re.split(r'(\*\*.*?\*\*|\*.*?\*)', cell_text)
                                for token in tokens:
                                    if not token:
                                        continue
                                    if token.startswith('**') and token.endswith('**'):
                                        run = p.add_run(token[2:-2])
                                        run.bold = True
                                    elif token.startswith('*') and token.endswith('*'):
                                        run = p.add_run(token[1:-1])
                                        run.italic = True
                                    else:
                                        run = p.add_run(token)
                                        
                                    if is_header:
                                        run.font.bold = True
                                        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                                    else:
                                        run.font.size = Pt(10)
            continue

        # Bullet List Items
        if stripped.startswith('- ') or stripped.startswith('* '):
            add_styled_paragraph(doc, stripped[2:], style='List Bullet', space_before=1, space_after=3)
            i += 1
            continue

        # Numbered List Items
        if re.match(r'^\d+\.\s', stripped):
            item_text = re.sub(r'^\d+\.\s', '', stripped)
            add_styled_paragraph(doc, item_text, style='List Number', space_before=1, space_after=3)
            i += 1
            continue

        # Figure Captions / Italics
        if stripped.startswith('*Figure ') and stripped.endswith('*'):
            p = add_styled_paragraph(doc, stripped[1:-1], space_before=4, space_after=12)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.italic = True
                run.font.size = Pt(9.5)
                run.font.color.rgb = RGBColor(0x4A, 0x55, 0x68)
            i += 1
            continue

        # Standard Paragraph
        add_styled_paragraph(doc, stripped, space_before=0, space_after=6)
        i += 1
        
    doc.save(DOCX_PATH)
    print(f"Successfully generated DOCX paper: {DOCX_PATH}")

if __name__ == "__main__":
    create_docx_from_markdown()
