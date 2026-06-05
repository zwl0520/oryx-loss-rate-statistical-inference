"""
论文格式修复脚本
===============
1. 将 Markdown 正确转换为 HTML（MathJax 渲染数学公式）
2. 修复 PDF 所需的 LaTeX 数学模式包裹
"""

import sys, re, os
sys.stdout.reconfigure(encoding='utf-8')

with open('paper/T1_论文.md', 'r', encoding='utf-8') as f:
    content = f.read()

# ===========================
# 修复1: 将行外希腊/数学符号包裹在 $...$ 中
# ===========================
# 这些字符在 LaTeX 文本模式下不存在，必须在数学模式中
fixes = {
    'λ': r'$\lambda$',
    'λ̂': r'$\hat{\lambda}$',
    'χ²': r'$\chi^2$',
    '≥': r'$\geq$',
    '≤': r'$\leq$',
    'α': r'$\alpha$',
    'β': r'$\beta$',
    'Φ': r'$\Phi$',
    'ρ': r'$\rho$',
    'θ̂': r'$\hat{\theta}$',
}

# 只在非数学模式中替换（不在已有的 $...$ 或 $$...$$ 中）
# 简单方法：对每行，在 $ 之外的区域替换
lines = content.split('\n')
fixed_lines = []
for line in lines:
    # 分割为数学模式内/外的片段
    parts = line.split('$')
    for i in range(0, len(parts), 2):  # 偶数索引 = 数学模式外
        if i < len(parts):
            for char, replacement in fixes.items():
                # 只替换不在已有数学模式中的
                parts[i] = parts[i].replace(char, replacement)
    fixed_lines.append('$'.join(parts))

fixed_content = '\n'.join(fixed_lines)

# 保存PDF兼容版
with open('paper/T1_论文_pdf.md', 'w', encoding='utf-8') as f:
    f.write(fixed_content)

# ===========================
# 修复2: 生成正确渲染的 HTML
# ===========================
import markdown

# 配置 markdown 扩展
md = markdown.Markdown(extensions=[
    'tables',          # 表格支持
    'fenced_code',     # 代码块
    'codehilite',      # 代码高亮
    'toc',             # 目录
    'nl2br',           # 换行
])

# 转换 markdown → HTML body
html_body = md.convert(fixed_content)

# 修复代码块中的问题：markdown 可能把数学模式中的字符也转换了
# 把 $...$ 和 $$...$$ 转换为 MathJax 兼容格式（已经是了，保持不变）

html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>俄乌装备损失率的"公平比较"——从估计偏误到区间估计</title>
<script>
MathJax = {{
  tex: {{
    inlineMath: [["$", "$"]],
    displayMath: [["$$", "$$"]],
    processEscapes: true,
  }}
}};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
  body {{
    max-width: 860px;
    margin: 30px auto;
    padding: 20px 40px;
    font-family: "Microsoft YaHei", "SimHei", "PingFang SC", sans-serif;
    font-size: 15px;
    line-height: 1.85;
    color: #222;
    background: #fff;
  }}
  h1 {{
    text-align: center;
    font-size: 22px;
    border-bottom: 2px solid #2c3e50;
    padding-bottom: 12px;
    margin-bottom: 10px;
  }}
  h2 {{
    font-size: 18px;
    border-bottom: 1px solid #bdc3c7;
    padding-bottom: 6px;
    margin-top: 28px;
    color: #2c3e50;
  }}
  h3 {{ font-size: 16px; color: #34495e; margin-top: 20px; }}
  h4 {{ font-size: 15px; color: #555; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 14px 0;
    font-size: 14px;
  }}
  th, td {{
    border: 1px solid #ccc;
    padding: 7px 10px;
    text-align: center;
  }}
  th {{ background: #ecf0f1; font-weight: bold; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  pre {{
    background: #f5f5f5;
    padding: 12px 16px;
    overflow-x: auto;
    font-size: 13px;
    border-radius: 4px;
    border: 1px solid #e0e0e0;
  }}
  code {{
    background: #f0f0f0;
    padding: 1px 5px;
    border-radius: 3px;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 90%;
  }}
  pre code {{ background: none; padding: 0; }}
  blockquote {{
    border-left: 4px solid #3498db;
    margin: 14px 0;
    padding: 6px 16px;
    background: #f0f7fb;
    color: #555;
  }}
  hr {{ border: none; border-top: 1px solid #ddd; margin: 28px 0; }}
  p {{ text-align: justify; }}
  strong {{ color: #2c3e50; }}
  em {{ font-style: italic; }}
  .toc {{ background: #f8f9fa; padding: 15px 20px; border-radius: 6px; margin-bottom: 20px; }}
</style>
</head>
<body>
{html_body}
</body>
</html>'''

with open('paper/T1_论文.html', 'w', encoding='utf-8') as f:
    f.write(html)

print(f'✓ paper/T1_论文.html 已生成 ({len(html):,} bytes)')
print(f'✓ paper/T1_论文_pdf.md 已生成 ({len(fixed_content):,} bytes)')
print(f'  修复了 {sum(fixed_content.count(c) for c in fixes)} 处数学符号包裹')
