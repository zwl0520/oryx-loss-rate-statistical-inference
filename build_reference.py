#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a clean reference PPTX for pandoc with Chinese font and color theme"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn, nsmap
from lxml import etree
import os

FONT = 'Microsoft YaHei'
DARK_BLUE = '1B2A4A'
BLUE = '2C5F8A'
LIGHT_BLUE = '3A7CBF'
WHITE = 'FFFFFF'
DARK_GRAY = '333333'
MED_GRAY = '7F8C8D'

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

def inject_font_xml(slide_or_layout, element_tag='p:cSld'):
    """Inject default font settings into slide XML"""
    cSld = slide_or_layout.element.find(qn(element_tag))
    if cSld is None:
        return

    # Build spTree if needed
    for spTree in cSld.findall(qn('p:spTree')):
        # Add a:lstStyle with default fonts
        lstStyle = etree.SubElement(spTree, qn('a:lstStyle'))
        defRPr = etree.SubElement(lstStyle, qn('a:defRPr'))
        defRPr.set('sz', '1800')
        latin = etree.SubElement(defRPr, qn('a:latin'))
        latin.set('typeface', FONT)
        ea = etree.SubElement(defRPr, qn('a:ea'))
        ea.set('typeface', FONT)
        break

# Apply to each slide layout and set colors/fonts
for layout in prs.slide_masters[0].slide_layouts:
    layout_name = layout.name
    # Set background color to light
    try:
        bg = layout.background
        # We'll leave backgrounds as-is and handle per-slide in pandoc
    except:
        pass

# Save reference template
tmpl_path = 'output/ref.pptx'
os.makedirs('output', exist_ok=True)
prs.save(tmpl_path)

# Now manually fix the XML to add proper font defaults
# This is the key step - we inject East Asian font defaults
import zipfile
import shutil
from io import BytesIO

# Read saved template, modify XML inside
def patch_pptx_fonts(pptx_path, output_path):
    """Patch PPTX to set default fonts in theme and masters"""
    tmp_path = pptx_path + '.tmp'

    with zipfile.ZipFile(pptx_path, 'r') as zin:
        with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)

                # Patch theme XML
                if item.filename.endswith('.xml') and 'theme' in item.filename.lower():
                    try:
                        root = etree.fromstring(data)
                        # Set default fonts in theme
                        for defRPr in root.iter(qn('a:defRPr')):
                            # Remove existing and re-add
                            for tag in [qn('a:latin'), qn('a:ea'), qn('a:cs')]:
                                for el in defRPr.findall(tag):
                                    defRPr.remove(el)
                            latin = etree.SubElement(defRPr, qn('a:latin'))
                            latin.set('typeface', FONT)
                            ea = etree.SubElement(defRPr, qn('a:ea'))
                            ea.set('typeface', FONT)
                        # Also fix majorFont and minorFont in theme
                        for elem in root.iter():
                            if elem.tag in [qn('a:majorFont'), qn('a:minorFont')]:
                                for tag in [qn('a:latin'), qn('a:ea'), qn('a:cs')]:
                                    for el in elem.findall(tag):
                                        el.set('typeface', FONT)
                        data = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
                    except:
                        pass

                # Patch slide master/layout XML
                if 'slideMaster' in item.filename or 'slideLayout' in item.filename:
                    try:
                        root = etree.fromstring(data)
                        # Set fonts on text defaults
                        for elem in root.iter():
                            for rPr in elem.findall(qn('a:rPr')):
                                for tag in [qn('a:latin'), qn('a:ea')]:
                                    for el in rPr.findall(tag):
                                        el.set('typeface', FONT)
                            for endParaRPr in elem.findall(qn('a:endParaRPr')):
                                for tag in [qn('a:latin'), qn('a:ea')]:
                                    for el in endParaRPr.findall(tag):
                                        el.set('typeface', FONT)
                            for defRPr in elem.iter(qn('a:defRPr')):
                                for tag in [qn('a:latin'), qn('a:ea')]:
                                    for el in defRPr.findall(tag):
                                        el.set('typeface', FONT)
                        data = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
                    except:
                        pass

                # Patch presentation.xml for default text styles
                if item.filename == 'ppt/presentation.xml':
                    try:
                        root = etree.fromstring(data)
                        for defRPr in root.iter(qn('a:defRPr')):
                            for tag in [qn('a:latin'), qn('a:ea')]:
                                for el in defRPr.findall(tag):
                                    el.set('typeface', FONT)
                        # Also fix defaultTextStyle
                        for dts in root.iter(qn('p:defaultTextStyle')):
                            for defRPr in dts.iter(qn('a:defRPr')):
                                for tag in [qn('a:latin'), qn('a:ea')]:
                                    for el in defRPr.findall(tag):
                                        el.set('typeface', FONT)
                        data = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
                    except:
                        pass

                zout.writestr(item, data)

    shutil.move(tmp_path, output_path)

patch_pptx_fonts(tmpl_path, tmpl_path)
print(f"Reference template: {tmpl_path}")
