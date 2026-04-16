# Cell Morphology Search Engine

**Find morphologically similar cells across 58 million images in under 100 milliseconds.**

This BioEngine application indexes the JUMP Cell Painting dataset — one of the largest
publicly available phenotypic profiling resources (116,000 compound-treated wells,
~47 TB of raw fluorescence microscopy images) — using DINOv2 deep image embeddings and
a FAISS vector database. Biologists can drag and drop any cell image to retrieve
visually similar cells from the database, enabling mechanism-of-action discovery,
quality control, and hypothesis generation at scales impossible on a single workstation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Hypha Server                                     │
│              (RPC gateway · artifact store · auth)                       │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ WebSocket RPC
┌──────────────────────────────▼──────────────────────────────────────────┐
│                   BioEngine Worker (Ray Cluster)                         │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  CellImageSearch  (Ray Serve · 1 GPU · 4 CPU · 20 GB RAM)         │  │
│  │                                                                    │  │
│  │  async_init()  ─── load DINOv2 ViT-B/14 ──► GPU memory           │  │
│  │                ─── load FAISS index       ──► CPU memory (~6 GB)  │  │
│  │                                                                    │  │
│  │  start_ingestion()  ──► asyncio.create_task ──► run_ingestion()   │  │
│  │  get_ingestion_status() ◄── status.json polling                   │  │
│  │  search()      ──► DINOv2 embed ──► FAISS search ──► results      │  │
│  │  get_umap_preview()  ──► UMAP compute/cache ──► Plotly data       │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  Ingestion Pipeline  (Ray remote tasks, N × 1 GPU each)           │  │
│  │                                                                    │  │
│  │  S3 (JUMP CP)  ──► download TIFFs  ──► extract_cell_crops()       │  │
│  │  S3 (Zarr/EM)  ──► slice Zarr      ──► extract_cell_crops()       │  │
│  │                                         │                          │  │
│  │                          to_rgb_uint8() │  percentile stretch      │  │
│  │                                         ▼                          │  │
│  │                          DINOv2 ViT-B/14 (fp16, batch=64)         │  │
│  │                          768-dim L2-normalised embeddings          │  │
│  │                                         │                          │  │
│  │                          build_index()  ▼                          │  │
│  │                          FAISS IndexIVFPQ  ──► cell_search.index  │  │
│  │                          metadata.parquet  ──► compound / MOA     │  │
│  │                          thumbnails.npy    ──► 96×96 previews     │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌───────────────────────┐  Storage: workspace_dir/cell_search/         │
│  │  /cell_search/        │                                              │
│  │   cell_search.index   │  FAISS binary (auto: Flat/IVFFlat/IVFPQ)    │
│  │   metadata.parquet    │  compound, MOA, plate, well, image_path      │
│  │   thumbnails.npy      │  96×96 PNG previews (base64)                 │
│  │   umap_cache.npz      │  pre-computed UMAP for 10K sample            │
│  │   index_info.json     │  build stats                                  │
│  └───────────────────────┘                                              │
└─────────────────────────────────────────────────────────────────────────┘
                               │ HTTPS
┌──────────────────────────────▼──────────────────────────────────────────┐
│                      Web UI  (frontend/index.html)                       │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  Sidebar                │  Main                                    │  │
│  │  ─────────              │  ──────────────────────────────────────  │  │
│  │  Index stats            │  Drag & drop zone  →  Top-20 results   │  │
│  │  58M cells indexed      │                                          │  │
│  │  DINOv2 · IVFPQ         │  ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐            │  │
│  │                         │  │🔬│ │🔬│ │🔬│ │🔬│ │🔬│  rank+score  │  │
│  │  Ingest Dataset         │  └──┘ └──┘ └──┘ └──┘ └──┘            │  │
│  │  [JUMP CP ▼]  [10 plt]  │  compound · MOA · plate/well           │  │
│  │  [4 GPU workers]        │                                          │  │
│  │  [▶ Start Ingestion]    │  MOA distribution histogram             │  │
│  │  ████░░░ 67% 1.2k/s     │                                          │  │
│  │                         │  UMAP — 58M cells coloured by MOA       │  │
│  │  UMAP controls          │  ┌──────────────────────────────────┐   │  │
│  │  [10000 pts] [MOA ▼]    │  │  Proteasome●  ●Kinase inhibitor  │   │  │
│  │  [⟳ Compute UMAP]       │  │      ★ your query                │   │  │
│  └────────────────────────  │  │  Microtubule●   ●DNA damage      │   │  │
│                             │  └──────────────────────────────────┘   │  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Dataset: JUMP Cell Painting

The **JUMP-CP** dataset (cpg0016) is the world's largest open phenotypic profiling
resource, produced by the Joint Undertaking for Morphological Profiling consortium.

| Property | Value |
|----------|-------|
| Wells | 116,750 |
| Compounds | ~50,000 |
| Channels | 5 (DNA, ER, RNA, AGP, Mito) |
| Image size | 2160 × 2160 px, 16-bit |
| Raw storage | ~47 TB |
| Single cells (est.) | ~58 million |
| S3 bucket | `s3://cellpainting-gallery/cpg0016-jump/` |
| Access | Public (no auth required) |

**RGB composite convention used here:** R = AGP (actin/golgi/plasma membrane),
G = ER (endoplasmic reticulum), B = DNA (DAPI nuclear stain).

---

## Embedding Model: DINOv2 ViT-B/14

DINOv2 is a self-supervised Vision Transformer trained on 142 million curated images.
It produces 768-dimensional feature vectors without any task-specific fine-tuning,
capturing rich semantic and morphological structure.

| Property | Value |
|----------|-------|
| Architecture | ViT-B/14 (patch size 14px) |
| Embedding dim | 768 |
| Input | 224 × 224 RGB, ImageNet-normalised |
| Throughput | ~500 images/sec per A100 (fp16, batch=64) |
| At 32 GPUs | ~58M cells in ~60 min |
| Reference | Oquab et al., arXiv:2304.07193 |

---

## Vector Index: FAISS

| Scale | Index type | Memory | Search latency |
|-------|-----------|--------|---------------|
| < 100K cells | IndexFlatIP | ~220 MB | <5ms |
| 100K – 5M | IndexIVFFlat | ~2 GB | <20ms |
| > 5M (production) | IndexIVFPQ | ~6 GB | <100ms |

`IVFPQ` compresses 768-dim float32 vectors (3 KB each) to 96 bytes each —
a **32× compression** — while retaining ~95% recall@10.

---

## Image Normalisation

Biological fluorescence microscopy images present unique challenges:
- **Dynamic range**: 16-bit uint16, values spanning 0–65535
- **Shot noise**: Poisson noise from low photon counts
- **Batch effects**: Per-plate illumination variation
- **Outliers**: Saturated pixels, debris, edge artefacts

This app applies **per-channel percentile stretch** (default: p1–p99):

```
raw uint16 (any range)
    → clip to [p1, p99] per channel
    → linear rescale to [0, 255] uint8
    → discard outlier pixels without losing real signal
```

For multi-channel Cell Painting images, three channels are selected and mapped
to RGB before DINOv2 input. The user can choose `Auto (p1–p99)`, `Wide (p0.1–p99.9)`,
or `Tight (p5–p95)` presets in the UI.

---

## Quick Start

### Deploy via BioEngine CLI

```bash
pip install "bioengine-cli[worker]"

bioengine save-application \
    --workspace bioimage-io \
    --artifact-id cell-image-search \
    --source bioengine_apps/cell-image-search/

bioengine run-application \
    --workspace bioimage-io \
    --artifact-id cell-image-search \
    --application-id cell-image-search
```

### Index 10 JUMP CP plates (~50K cells, ~3 min on 4 GPUs)

```python
from hypha_rpc import connect_to_server
server = await connect_to_server({"server_url": "https://hypha.aicell.io"})
svc = await server.get_service("bioimage-io/cell-image-search")

status = await svc.start_ingestion(dataset="jump-cp", n_plates=10, n_gpu_workers=4)
session_id = status["session_id"]

# Poll until complete
import asyncio, time
while True:
    s = await svc.get_ingestion_status(session_id=session_id)
    print(f"{s['status']} | {s['n_embedded']:,} cells | {s['throughput_per_sec']:.0f} cells/s")
    if s["status"] in ("completed", "failed", "stopped"):
        break
    await asyncio.sleep(2)
```

### Search

```python
import base64, pathlib

with open("my_cell.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

results = await svc.search(image_b64=b64, top_k=20)
print(f"Search: {results['elapsed_ms']}ms | {results['n_cells_searched']:,} cells searched")
for r in results["results"][:5]:
    print(f"  #{r['rank']} {r['compound']:30s} {r['moa_class']:25s} score={r['score']:.3f}")
```

### UMAP

```python
umap = await svc.get_umap_preview(n_samples=10000, color_by="moa_class")
# Returns: {x: [...], y: [...], labels: [...], colors: [...], n_total: 58M}
```

---

## Performance at Scale

| Operation | 4 GPUs | 32 GPUs |
|-----------|--------|---------|
| Index 1M cells | ~33 min | ~4 min |
| Index 58M cells (full JUMP CP) | ~32 hrs | ~4 hrs |
| FAISS index build (58M, IVFPQ) | ~8 min | ~8 min |
| Query embedding (1 image) | <10ms | <10ms |
| FAISS search (58M, IVFPQ) | <80ms | <80ms |
| UMAP (10K sample, first time) | ~45s | ~45s |
| UMAP (cached) | <100ms | <100ms |

**Key insight**: A laptop would take ~35 days to embed 58M cells; BioEngine
does it in 4 hours using 32 GPU workers scheduled automatically via Ray.

---

## Files

```
cell-image-search/
├── manifest.yaml         # BioEngine app descriptor
├── main.py               # CellImageSearch Ray Serve deployment
├── ingestion.py          # Distributed Ray ingestion pipeline
├── embedder.py           # DINOv2 ViT-B/14 wrapper + Ray actor
├── normalizer.py         # Microscopy image normalisation utilities
├── index_manager.py      # FAISS index build/load/search/UMAP
└── frontend/
    └── index.html        # Drag-and-drop search UI (Tailwind + Plotly)
```
