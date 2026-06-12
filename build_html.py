#!/usr/bin/env python3
"""
将项目中的 Markdown 文件转换为内嵌 MathJax 的 HTML 文件
确保数学公式在任何浏览器中都能正确渲染
"""

import sys, re, os

sys.stdout.reconfigure(encoding='utf-8')

# ===========================
# KaTeX 兼容性修复
# ===========================
def fix_katex_compatibility(content):
    """修复 KaTeX/VS Code Markdown 预览不兼容的 LaTeX 命令"""
    fixes = {
        # \stackrel 在部分 KaTeX 版本中不被支持 → \overset
        r'\stackrel{iid}{\sim}': r'\overset{\text{iid}}{\sim}',
        r'\stackrel{\cdot}{\sim}': r'\overset{\cdot}{\sim}',
        r'\stackrel{d}{\rightarrow}': r'\xrightarrow{d}',
    }
    for old, new in fixes.items():
        content = content.replace(old, new)
    return content


# ===========================
# Markdown → HTML 转换 (轻量级，保留 GFM 特性)
# ===========================
def md_to_html(content):
    """简单但可靠的 Markdown → HTML 转换，特别处理数学公式"""
    lines = content.split('\n')
    html_lines = []
    in_code_block = False
    in_table = False
    in_list = False
    in_blockquote = False
    table_rows = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # 代码块
        if line.strip().startswith('```'):
            if in_code_block:
                html_lines.append('</code></pre>')
                in_code_block = False
            else:
                lang = line.strip()[3:].strip()
                html_lines.append(f'<pre><code class="language-{lang}">')
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            html_lines.append(escape_html(line))
            i += 1
            continue

        # 允许论文图题使用受控 HTML，不转义为文本
        stripped = line.strip()
        if stripped.startswith('<p align="center"><strong>图') and stripped.endswith('</p>'):
            html_lines.append(stripped)
            i += 1
            continue

        # 显示公式块：Markdown 中使用单独的 $$ 起止行，HTML 中交给 MathJax 渲染
        if stripped == '$$':
            math_lines = ['$$']
            i += 1
            while i < len(lines):
                math_lines.append(lines[i].strip())
                if lines[i].strip() == '$$':
                    i += 1
                    break
                i += 1
            html_lines.append('<div class="math-block">')
            html_lines.extend(math_lines)
            html_lines.append('</div>')
            continue

        # 表格 (需要收集所有行)
        if '|' in line and line.strip().startswith('|'):
            if not in_table:
                in_table = True
                table_rows = []
            table_rows.append(line)
            # 检查下一行是否还是表格
            if i + 1 < len(lines) and '|' in lines[i + 1] and lines[i + 1].strip().startswith('|'):
                i += 1
                continue
            else:
                # 输出整个表格
                html_lines.append(render_table(table_rows))
                table_rows = []
                in_table = False
                i += 1
                continue

        # 标题
        if line.startswith('#### '):
            html_lines.append(f'<h4>{process_inline(line[5:])}</h4>')
        elif line.startswith('### '):
            html_lines.append(f'<h3>{process_inline(line[4:])}</h3>')
        elif line.startswith('## '):
            html_lines.append(f'<h2>{process_inline(line[3:])}</h2>')
        elif line.startswith('# '):
            html_lines.append(f'<h1>{process_inline(line[2:])}</h1>')

        # 水平线
        elif line.strip() in ('---', '***', '___'):
            html_lines.append('<hr>')

        # 引用
        elif line.startswith('> '):
            content_part = line[2:]
            if not in_blockquote:
                html_lines.append('<blockquote>')
                in_blockquote = True
            html_lines.append(f'<p>{process_inline(content_part)}</p>')
            if i + 1 >= len(lines) or not lines[i + 1].startswith('> '):
                html_lines.append('</blockquote>')
                in_blockquote = False

        # 无序列表
        elif re.match(r'^[\s]*[-*+]\s', line):
            indent = len(line) - len(line.lstrip())
            content_part = re.sub(r'^[\s]*[-*+]\s', '', line)
            if not in_list:
                html_lines.append('<ul>')
                in_list = True
            html_lines.append(f'<li>{process_inline(content_part)}</li>')
            if i + 1 >= len(lines) or not re.match(r'^[\s]*[-*+]\s', lines[i + 1]):
                html_lines.append('</ul>')
                in_list = False

        # 有序列表
        elif re.match(r'^[\s]*\d+\.\s', line):
            content_part = re.sub(r'^[\s]*\d+\.\s', '', line)
            if not in_list:
                html_lines.append('<ol>')
                in_list = True
            html_lines.append(f'<li>{process_inline(content_part)}</li>')
            if i + 1 >= len(lines) or not re.match(r'^[\s]*\d+\.\s', lines[i + 1]):
                html_lines.append('</ol>')
                in_list = False

        # 空行
        elif line.strip() == '':
            if in_list:
                html_lines.append('</ul>' if '<ul>' in html_lines[-5:] else '</ol>')
                in_list = False
            html_lines.append('')

        # 普通段落
        else:
            html_lines.append(f'<p>{process_inline(line)}</p>')

        i += 1

    return '\n'.join(html_lines)


def escape_html(text):
    """转义 HTML 特殊字符（但保留 math $ 分隔符）"""
    # 保护数学公式
    math_blocks = []
    def save_math(m):
        math_blocks.append(m.group(0))
        return f'\x00MATH{len(math_blocks)-1}\x00'

    # 保存 $$...$$ 块
    text = re.sub(r'\$\$[^$]+\$\$', save_math, text)
    # 保存 $...$ 内联
    text = re.sub(r'\$[^$]+\$', save_math, text)

    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    # 恢复数学公式
    for i, block in enumerate(math_blocks):
        text = text.replace(f'\x00MATH{i}\x00', block)

    return text


def process_inline(text):
    """处理内联元素：加粗、斜体、代码、链接、图片、数学公式"""
    text = escape_html(text)

    # 图片 ![alt](url)
    text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'<img src="\2" alt="\1" style="max-width:100%">', text)
    # 链接 [text](url)
    text = re.sub(r'\[([^\]]*)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
    # 行内代码 `...`
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    # 加粗 **...**
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    # 斜体 *...*
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<em>\1</em>', text)
    # 删除线 ~~...~~
    text = re.sub(r'~~([^~]+)~~', r'<del>\1</del>', text)

    return text


def render_table(rows):
    """渲染 Markdown 表格为 HTML"""
    if len(rows) < 2:
        return ''

    # 跳过第二行（分隔符行）
    header = rows[0]
    data_rows = rows[2:] if len(rows) > 2 else []

    # 解析单元格
    def parse_cells(row):
        cells = [c.strip() for c in row.split('|')]
        return [c for c in cells if c]  # 去掉首尾空

    headers = parse_cells(header)

    html = '<table>\n<thead>\n<tr>\n'
    for h in headers:
        html += f'<th>{process_inline(h)}</th>\n'
    html += '</tr>\n</thead>\n<tbody>\n'

    for row in data_rows:
        cells = parse_cells(row)
        html += '<tr>\n'
        for c in cells:
            html += f'<td>{process_inline(c)}</td>\n'
        html += '</tr>\n'

    html += '</tbody>\n</table>'
    return html


# ===========================
# HTML 模板
# ===========================
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script>
MathJax = {{
  tex: {{
    inlineMath: [["$", "$"]],
    displayMath: [["$$", "$$"]],
    processEscapes: true,
    tags: "ams",
  }},
  options: {{
    ignoreHtmlClass: "no-math",
    processHtmlClass: "math",
  }}
}};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
  body {{
    max-width: 900px;
    margin: 30px auto;
    padding: 20px 40px;
    font-family: "Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans SC", sans-serif;
    font-size: 15px;
    line-height: 1.9;
    color: #222;
    background: #fff;
  }}
  h1 {{
    text-align: center;
    font-size: 24px;
    border-bottom: 3px solid #1B2A4A;
    padding-bottom: 14px;
    margin-bottom: 12px;
    color: #1B2A4A;
  }}
  h2 {{
    font-size: 20px;
    border-bottom: 1.5px solid #bdc3c7;
    padding-bottom: 8px;
    margin-top: 32px;
    color: #2C5F8A;
  }}
  h3 {{ font-size: 17px; color: #34495e; margin-top: 24px; }}
  h4 {{ font-size: 15px; color: #555; margin-top: 18px; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 16px 0;
    font-size: 14px;
  }}
  th, td {{
    border: 1px solid #ccc;
    padding: 8px 12px;
    text-align: center;
  }}
  th {{ background: #ecf0f1; font-weight: bold; color: #1B2A4A; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  pre {{
    background: #f5f5f5;
    padding: 14px 18px;
    overflow-x: auto;
    font-size: 13px;
    border-radius: 6px;
    border: 1px solid #e0e0e0;
    line-height: 1.5;
  }}
  code {{
    background: #f0f0f0;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: "Consolas", "Courier New", "Fira Code", monospace;
    font-size: 90%;
  }}
  pre code {{ background: none; padding: 0; }}
  blockquote {{
    border-left: 4px solid #2C5F8A;
    margin: 16px 0;
    padding: 8px 18px;
    background: #f0f7fb;
    color: #555;
  }}
  hr {{ border: none; border-top: 1px solid #ddd; margin: 32px 0; }}
  p {{ text-align: justify; }}
  p[align="center"] {{ text-align: center; text-indent: 0; }}
  .math-block {{ text-align: center; margin: 1em 0; }}
  p:has(> mjx-container[display="true"]) {{ text-align: center; text-indent: 0; }}
  mjx-container[display="true"] {{
    display: block;
    text-align: center !important;
    margin: 1em 0;
  }}
  strong {{ color: #1B2A4A; }}
  ul, ol {{ margin: 12px 0; padding-left: 28px; }}
  li {{ margin: 4px 0; }}
  a {{ color: #2C5F8A; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  img {{ max-width: 100%; height: auto; border-radius: 4px; }}
  .toc {{ background: #f8f9fa; padding: 18px 24px; border-radius: 8px; margin-bottom: 24px; border: 1px solid #e8e8e8; }}
  @media (max-width: 768px) {{
    body {{ padding: 12px 16px; font-size: 14px; }}
    table {{ font-size: 12px; }}
  }}
</style>
</head>
<body>
{body}
</body>
</html>'''


# ===========================
# 主转换函数
# ===========================
def convert_file(md_path, html_path, title=None):
    """转换单个 Markdown 文件为 HTML"""
    print(f"转换: {md_path} → {html_path}")

    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 修复 KaTeX 兼容性
    content = fix_katex_compatibility(content)

    # 提取标题（第一个 # 开头的行）
    if title is None:
        for line in content.split('\n'):
            if line.startswith('# '):
                title = line[2:].strip()
                break
    if title is None:
        title = os.path.basename(md_path)

    # 转换为 HTML
    body = md_to_html(content)

    # 填充模板
    html = HTML_TEMPLATE.format(title=title, body=body)

    # 写入文件
    os.makedirs(os.path.dirname(html_path) if os.path.dirname(html_path) else '.', exist_ok=True)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)

    size_kb = len(html) / 1024
    print(f"  ✓ {html_path} ({size_kb:.0f} KB)")
    return html_path


# ===========================
# 批量转换
# ===========================
if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))

    files_to_convert = [
        ('README.md', 'output/html/README.html', '俄乌装备损失率统计推断 — 项目 README'),
        ('PROJECT_GUIDE.md', 'output/html/PROJECT_GUIDE.html', '俄乌装备损失率统计分析 — 项目完全指南'),
        ('slides.md', 'output/html/slides.html', '俄乌装备损失率的公平比较 — 答辩幻灯片'),
        ('output/展示讲稿.md', 'output/html/展示讲稿.html', '俄乌装备损失率的公平比较 — 展示讲稿'),
    ]

    print("=" * 60)
    print("Markdown → HTML (MathJax) 批量转换")
    print("=" * 60)

    for md_rel, html_rel, title in files_to_convert:
        md_path = os.path.join(base_dir, md_rel)
        html_path = os.path.join(base_dir, html_rel)
        if os.path.exists(md_path):
            convert_file(md_path, html_path, title)
        else:
            print(f"  ⚠ 跳过 (文件不存在): {md_path}")

    print(f"\n所有 HTML 文件保存至: {os.path.join(base_dir, 'output', 'html')}/")
    print("用浏览器打开这些 HTML 文件即可看到完美渲染的数学公式。")
