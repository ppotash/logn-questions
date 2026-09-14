import sys, json, glob, io, hashlib
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from logn import prompts
from logn.game import load_corpus
from pathlib import Path

pool, man = load_corpus(Path('.'))
vid = next(i for i, d in pool.items() if d['title'] == 'Grupo Vocento')
print(f"Grupo Vocento = {vid}\n")

key = lambda q, d: hashlib.sha256(f"{d}\x00{q}".encode()).hexdigest()[:20]
judges = {}
for f in Path('results/adjudicated').glob('judgments*.json'):
    judges[f.stem.replace('judgments_', '')] = json.loads(f.read_text(encoding='utf-8'))

rows = []
for f in glob.glob('results/raw/*/*/*.json'):
    g = json.load(io.open(f, encoding='utf-8'))
    if g.get('aborted') or g['target_id'] != vid:
        continue
    if g['prompt_version'] != prompts.PROMPT_VERSION:
        continue
    for h in g['history']:
        v = [judges[j].get(key(h['question'], vid)) for j in judges]
        rows.append((g['model'], g['size'], h['round'], h['answer'], v, h['question']))

print(f"{len(rows)} logged questions about Vocento under prompt "
      f"{prompts.PROMPT_VERSION}\n")
bad = [r for r in rows if any(x is not None and x != r[3] for x in r[4])]
print(f"{len(bad)} where at least one judge disagrees with the answer given:\n")
for m, n, rd, ans, v, q in bad:
    print(f"  {m:<17} N={n:<5} R{rd}  answered {'Yes' if ans else 'No':<3} "
          f"judges={['Y' if x else 'N' if x is not None else '?' for x in v]}")
    print(f"    {q[:110]}")