---
license: mit
pretty_name: HelioAI speasy parameter index
tags:
  - heliophysics
  - space-physics
  - speasy
  - embeddings
size_categories:
  - 10K<n<100K
---

# HelioAI speasy parameter index

The search index [HelioAI](https://github.com/erdoganfurkan/HelioAI) queries to find a
heliophysics data product: ~82 000 parameters from the archives
[speasy](https://github.com/SciQLop/speasy) reaches — AMDA, CDAWeb, the Cluster Science
Archive, SSCWeb — each described as text and embedded with
`sentence-transformers/all-MiniLM-L6-v2`.

`helioai index` downloads it on a fresh install instead of building it; there is no need
to fetch it by hand.

## How it is made

The `Index` workflow of the HelioAI repository builds it from scratch at every release tag
(`helioai index --rebuild`, no API key) and exports it with `helioai index --export`. The
tag of each release (`v0.4.0`, …) holds the index that release's code describes; `main`
holds the most recent one. A snapshot holding fewer than 95 % of the products expected is
refused rather than published: it means an archive did not answer during the build.

## Files

- `manifest.json` — format, HelioAI and speasy versions, embedding model, build date, and
  per collection the entry count, the vector dimension and a SHA-256 per file.
- `products.jsonl.gz`, `catalogs.jsonl.gz` — one JSON line per entry (`id`, `document`,
  `metadata`), in insertion order.
- `products.npy`, `catalogs.npy` — float32 embeddings, row *i* for line *i*.

## Sources and licence

The descriptions are the archives' own metadata as speasy exposes it, and remain theirs;
data obtained through them should be acknowledged as each archive asks — AMDA (CDPP),
CDAWeb and SSCWeb (NASA SPDF), the Cluster Science Archive (ESA). The index itself — its
assembly, the measurement type and region HelioAI adds to it, and the embeddings — is
released under the MIT licence, as HelioAI is.
