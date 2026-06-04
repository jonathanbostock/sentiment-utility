from __future__ import annotations

import random


def build_pool_sample(sources, quotas, n, seed=0):
    rng = random.Random(seed)
    seen = set()
    pooled = []
    chosen, meta = [], {}

    def add_unique(name, source, valence, bucket):
        key = name.strip().lower()
        if not name.strip() or key in seen:
            return
        seen.add(key)
        bucket.append((name, source, valence))

    per_source = {}
    for source, entries in sources.items():
        bucket = []
        for name, valence in entries:
            add_unique(name, source, valence, bucket)
        per_source[source] = bucket
        pooled.extend(bucket)

    for source, bucket in per_source.items():
        q = quotas.get(source, 0)
        picks = bucket if q >= len(bucket) else rng.sample(bucket, q)
        for name, src, val in picks:
            if name not in meta:
                chosen.append(name)
                meta[name] = {"source": src, "human_valence": val}

    if len(chosen) < n:
        remaining = [t for t in pooled if t[0] not in meta]
        rng.shuffle(remaining)
        for name, src, val in remaining:
            if len(chosen) >= n:
                break
            chosen.append(name)
            meta[name] = {"source": src, "human_valence": val}

    if len(chosen) > n:
        rng.shuffle(chosen)
        chosen = chosen[:n]
        meta = {k: meta[k] for k in chosen}

    return chosen, meta
