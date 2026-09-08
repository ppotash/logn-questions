# scripts/openers.py
import sys, json, glob, io
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

size = sys.argv[1] if len(sys.argv) > 1 else "1024"
runs = [int(x) for x in sys.argv[2:]] or [0, 1, 2]

by = {}
for f in glob.glob(f"results/raw/*/{size}/*.json"):
    g = json.load(io.open(f, encoding="utf-8"))
    if g.get("aborted") or not g["history"]:
        continue
    by.setdefault(g["model"], {})[g["run"]] = (
        g["history"][0]["question"], g["history"][0]["answer"], g["won"])

for r in runs:
    print(f"\n=== N={size}, run {r} ===")
    for m in sorted(by):
        if r in by[m]:
            q, a, w = by[m][r]
            print(f"  {m:<18} [{'Y' if a else 'N'}] {'W' if w else 'L'}  {q}")