# scripts/candidates.py
import glob, json
from collections import defaultdict
from math import log2

cells = defaultdict(list)
for f in glob.glob('results/raw/*/*/*.json'):
    g = json.load(open(f, encoding='utf-8'))
    if g.get('aborted'): continue
    for h in g['history']:
        # expected candidates entering this round, if prior splits were even
        cand = g['size'] / 2 ** (h['round'] - 1)
        cells[(g['model'], round(log2(cand)), h['round'] == 1)].append(h['answer'])

print(f"{'model':<18}{'log2(cand)':>11}{'round1?':>9}{'n':>5}{'yes':>7}")
for k in sorted(cells):
    v = cells[k]
    if len(v) >= 8:
        print(f"{k[0][:17]:<18}{k[1]:>11}{str(k[2]):>9}{len(v):>5}{sum(v)/len(v):>7.0%}")