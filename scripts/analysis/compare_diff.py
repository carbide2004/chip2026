import json

with open('submissions/qwen3_8b_zeroshot_A.jsonl', 'r', encoding='utf-8') as f:
    zs1 = [json.loads(l) for l in f]
with open('submissions/qwen3_8b_v2_optimized_A.jsonl', 'r', encoding='utf-8') as f:
    v2 = [json.loads(l) for l in f]

diff_ents = 0
for i in range(len(zs1)):
    n1 = {e['offset']: (e['text'], e.get('note')) for e in zs1[i]['entities']}
    n2 = {e['offset']: (e['text'], e.get('note')) for e in v2[i]['entities']}
    changed = []
    for off, (t, val) in n2.items():
        old_val = n1.get(off, (None, None))[1]
        if val != old_val:
            changed.append((t, old_val, val))
    if changed:
        diff_ents += 1
        print(f"Doc {i+1} PMC{v2[i]['pmc_id']}: {len(changed)} entities flipped note")
        for t, ov, nv in changed[:5]:
            print(f"   '{t}': {ov} -> {nv}")

print(f"\nTotal docs affected by negation rule change: {diff_ents}/20")
