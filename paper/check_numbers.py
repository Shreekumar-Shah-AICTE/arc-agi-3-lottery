"""List every number in paper/draft/*.md that is not in paper/results.json. Presence check only: a match on value
says nothing about meaning, so review still checks that each number comes from the right key. Skips HTML comments, headings,
citation brackets, section references and a short whitelist (years, SHA-256, discussion ids, GPU model, 0-9)."""
import json, re, glob
from pathlib import Path
R = Path(__file__).resolve().parents[1]
vals, times = set(), set()
for e in json.load(open(R / 'paper' / 'results.json')).values():
    v = e['value'].replace(',', '')
    if re.fullmatch(r'\d{1,2}:\d{2}', v): times.add(v); continue
    try: f = float(v)
    except ValueError: continue
    for g in (f, abs(f)):
        vals.update({g, round(g), round(g, 1), round(g, 2), round(100 * g, 1), round(100 * g)})
WHITE = {2018, 2019, 2021, 2024, 2025, 2026, 256, 6000, 755123, 755320, 3550011, 85}
bad = 0
for f in sorted(glob.glob(str(R / 'paper' / 'draft' / '*.md')) + glob.glob(str(R / 'paper' / 'writeup' / '*.md')) + glob.glob(str(R / 'paper' / 'tables' / 'tC_*.md')) + glob.glob(str(R / 'paper' / 'tables' / 'tD_*.md')) + glob.glob(str(R / 'paper' / 'tables' / 'tE_*.md'))):
    txt = re.sub(r'<!--.*?-->', '', open(f, encoding='utf-8').read(), flags=re.S)
    for ln, line in enumerate(txt.splitlines(), 1):
        if line.startswith('#'): continue
        t = re.sub(r'\[[^\]]*\]', '', line)
        t = re.sub(r'Sections? \d+(\.\d+)?(-\d+(\.\d+)?)?|Fig\. \d+|Appendix [A-Z]', '', t)
        for m in re.finditer(r'(?<![\w.:])(\d{1,2}:\d{2})(?![\w:])', t):  # clock times (UTC) must be registered as hh:mm
            if m.group(1) not in times: bad += 1; print(f'{Path(f).name}:{ln}: {m.group(0)}  | {line.strip()[:110]}')
        t = re.sub(r'(?<![\w.:])\d{1,2}:\d{2}(?![\w:])', ' ', t)
        for m in re.finditer(r'(?<![\w.])(\d[\d,]*(?:\.\d+)?)[kKM]?(?![\w])', t):
            x = float(m.group(1).replace(',', '').rstrip(','))
            if x in vals or x in WHITE or (x.is_integer() and x < 10): continue
            bad += 1; print(f'{Path(f).name}:{ln}: {m.group(0)}  | {line.strip()[:110]}')
print(f'{bad} unmatched numbers')
