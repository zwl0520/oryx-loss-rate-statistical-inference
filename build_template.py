#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a reference PPTX template for pandoc with Chinese font support"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
import copy
from lxml import etree

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

DARK_BLUE = RGBColor(0x1B, 0x2A, 0x4A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK_GRAY = RGBColor(0x33, 0x33, 0x33)
MED_GRAY = RGBColor(0x66, 0x66, 0x66)
ACCENT_BLUE = RGBColor(0x2C, 0x5F, 0x8A)
LIGHT_BG = RGBColor(0xF5, 0xF6, 0xFA)
FONT_FAMILY = 'Microsoft YaHei'

# We'll modify slide layouts for the SlideMaster
slide_master = prs.slide_masters[0]

# Helper to set font on runs in a placeholder
def set_font_on_shape(shape, font_name=FONT_FAMILY, font_size=None, color=None, bold=None):
    """Set font properties on all text in a shape"""
    if shape.has_text_frame:
        for para in shape.text_frame.paragraphs:
            para.font.name = font_name
            if font_size:
                para.font.size = Pt(font_size)
            if color:
                para.font.color.rgb = color
            if bold is not None:
                para.font.bold = bold
            for run in para.runs:
                run.font.name = font_name
                if font_size:
                    run.font.size = Pt(font_size)
                if color:
                    run.font.color.rgb = color
                if bold is not None:
                    run.font.bold = bold

# Modify each slide layout
for layout_idx, layout in enumerate(slide_master.slide_layouts):
    layout_name = layout.name if hasattr(layout, 'name') else f"Layout {layout_idx}"

    for ph in layout.placeholders:
        try:
            phf = ph.placeholder_format
            if phf.type == 1:  # TITLE (1)
                set_font_on_shape(ph, font_size=36, color=DARK_BLUE, bold=True)
            elif phf.type == 2:  # BODY (2)
                set_font_on_shape(ph, font_size=18, color=DARK_GRAY)
            elif phf.type == 3:  # CENTER_TITLE (3)
                set_font_on_shape(ph, font_size=32, color=DARK_BLUE, bold=True)
            elif phf.type == 4:  # SUBTITLE (4)
                set_font_on_shape(ph, font_size=20, color=MED_GRAY)
            elif phf.type == 7:  # FOOTER
                set_font_on_shape(ph, font_size=10, color=MED_GRAY)
            else:
                set_font_on_shape(ph, font_size=16, color=DARK_GRAY)
        except Exception:
            pass

# Also set default fonts in theme XML
# Access the theme
try:
    theme = slide_master.element.find('.//' + qn('a:theme'))
    if theme is None:
        # Try alternate path
        for part in prs.part.related_parts.values():
            xml = part.blob if hasattr(part, 'blob') else ''
except Exception:
    pass

# Set default paragraph font via XML manipulation
# This is the key to making Chinese fonts work in pandoc-generated PPTX
master_xml = slide_master.element

# Find or create default text styles
nsmap = {
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
}

# Add a:defaultTextStyle to theme if not present
# This is complex - let's use a simpler approach: set Latin and EastAsian fonts

def set_default_fonts(element, latin_font=FONT_FAMILY, ea_font=FONT_FAMILY):
    """Recursively set default fonts in XML tree"""
    for rPr in element.iter(qn('a:rPr')):
        # Latin font
        latin = rPr.find(qn('a:latin'))
        if latin is None:
            latin = etree.SubElement(rPr, qn('a:latin'))
        latin.set('typeface', latin_font)

        # East Asian font
        ea = rPr.find(qn('a:ea'))
        if ea is None:
            ea = etree.SubElement(rPr, qn('a:ea'))
        ea.set('typeface', ea_font)

    for defRPr in element.iter(qn('a:defRPr')):
        latin = defRPr.find(qn('a:latin'))
        if latin is None:
            latin = etree.SubElement(defRPr, qn('a:latin'))
        latin.set('typeface', latin_font)
        ea = defRPr.find(qn('a:ea'))
        if ea is None:
            ea = etree.SubElement(defRPr, qn('a:ea'))
        ea.set('typeface', ea_font)

# Apply to master
set_default_fonts(master_xml)

# Apply to each layout
for layout in slide_master.slide_layouts:
    set_default_fonts(layout.element)

# Save template
template_path = 'output/reference_template.pptx'
prs.save(template_path)
print(f"Template saved to {template_path}")
