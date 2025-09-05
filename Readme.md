# Whole‑Slide Imaging System (WSI) — Deep‑Learning Autofocus & Pyramidal OME‑TIFF

> Python · TensorFlow/PyTorch · OpenCV · NVIDIA Jetson · Dask/Zarr · OME‑TIFF

This repository contains core components of a Whole‑Slide Imaging pipeline built for high‑throughput digital pathology. It includes a deep‑learning autofocus regressor for inline focus estimation, a stitched whole‑slide image (WSI) assembly flow, and a memory/I/O‑optimized exporter that produces pyramidal OME‑TIFF suitable for viewers and downstream analysis.

---

## 🚀 Highlights

- **Deep‑Learning Autofocus (Regressor):** Learns continuous focus distance directly from patches, enabling **±0.002 mm** focus accuracy across multi‑resolution slides.
- **Seamless WSI Capture & Stitching:** Implements an **optimal capture strategy (patented)** to minimize seams and parallax while maximizing in‑focus coverage.
- **Edge‑Optimized Export:** Pyramidal OME‑TIFF generation with Dask/Zarr that reduces processing time and peak RAM footprint to fit **NVIDIA Jetson‑class** devices.
- **Modular, Inline Processing:** Designed to run *inline* during scanning: estimate focus → select sharp tiles → assemble → pyramidal export.
- **Rechunking for I/O Efficiency:** Smart chunk geometry for multi‑resolution tiling and fast random access in downstream viewers.

> **Note on IP:** Some capture/stitching logic is covered by a patent. Please respect the license/IP notes below before redistribution or commercialization.

---

## 🧭 Repository Snapshot

This ZIP contains a subset of the full project, focused on autofocus, accumulation utilities, and the OME‑TIFF pipeline.

```
Whole-Slide-Imaging-System-main/
├─ DL_model.py        # ResNet-based focus regressor + training/inference helpers
├─ img_acc.py         # Lightweight image/patch accumulator (double-ended queue)
├─ packagep2.py       # Dask/Zarr-backed WSI data adapter and exporter
└─ packages1.py       # Multiprocessing collection, raster/level registration, OME‑TIFF writer
```

### Key Modules & Classes

- **`ResNetFocusRegressor` (`DL_model.py`)**  
  - CNN regressor (ResNet backbone) with continuous focus prediction.  
  - Includes dataset/transform scaffolding (`InlineFocusDataset`), training loop (`train_inline_focus_model`), and inference (`predict_focus_distance`).

- **`d_que` (`img_acc.py`)**  
  - Minimal double‑ended queue for patch/window accumulation with `append`, `popleft`, `trim`, and random access.

- **`wsi_da` (`packagep2.py`)**  
  - Dask/Zarr data abstraction for WSI pyramids.  
  - Provides `rechucking`, `save_as_zarr`, and `zarr_to_ome_tiff` to move from intermediate arrays → Zarr → OME‑TIFF.

- **`ProcessCollection` (`packages1.py`)**  
  - Multiprocessing‑friendly collector for tile batches, with raster group registration and OME‑TIFF emission (`save_processed_images`).  
  - Static helper `zarr_process(queue, zarr_path, ome_tiff_path)` to run collection & export in a worker.

> Names and signatures above reflect the code in this archive; additional utilities (data loaders, device control, stitching heuristics, etc.) live in the full private repo.

---

## 🔬 Autofocus Model (Inline)

**Goal:** Predict the best‑focus offset from raw/near‑raw image patches so the stage can be driven proactively, reducing z‑stacking and capture time.

**Approach:**
- Supervised regression with a ResNet backbone (see `ResNetFocusRegressor`).
- Data pipeline in `InlineFocusDataset` with torchvision transforms.
- Learning‑rate scheduling via `ReduceLROnPlateau` for stable convergence.
- Exportable inference path `predict_focus_distance()` suitable for inline use.

**Why regression?**  
Continuous offsets preserve ordering information and allow sub‑step correction, which is critical for the reported **±0.002 mm** accuracy target.

---

## 🧵 Stitching & Capture Strategy (Patented)

The capture algorithm plans a path over the specimen with overlap tuned to the optics’ field curvature and depth of field. Inline focus scores select or re‑capture tiles that miss the sharpness threshold. Neighbor‑aware blending is applied during assembly to avoid visible seams.

> For IP‑protected pieces, this archive includes only the interfaces used by the exporter and accumulator.

---

## 🗂️ Pyramidal OME‑TIFF Export

- **Intermediate Format:** Zarr arrays (chunked, compressed) for scalable out‑of‑core processing.
- **Rechunking:** `wsi_da.rechucking()` optimizes chunk geometry per pyramid level for viewer access patterns.
- **Emission:** `wsi_da.zarr_to_ome_tiff()` writes standards‑compliant OME‑TIFF with multiple resolutions (pyramid levels).
- **Multiprocessing:** `ProcessCollection` gathers processed tiles from workers and persists them without exceeding Jetson memory budgets.

**Why OME‑TIFF?**  
Interoperable metadata + multi‑resolution pyramids → plug‑and‑play with common pathology viewers and analytics pipelines.

---

## 📈 Performance Notes

- **Focus Accuracy:** ±0.002 mm across multi‑resolution slides (validated inline on representative datasets).  
- **Throughput:** Pyramidal export via Dask/Zarr significantly reduces wall‑clock time compared to monolithic in‑memory assembly.  
- **Memory Footprint:** Peak RAM reduced to operate within **NVIDIA Jetson** constraints by streaming tiles and avoiding large intermediate buffers.

> Exact numbers depend on optics, camera, tile size, pyramid depth, compression, and overlap ratio.

---

## 🧩 Typical Pipeline

1. **Tile acquisition** with stage control and camera SDK.  
2. **Inline focus prediction** with `ResNetFocusRegressor` → select/correct.  
3. **Patch/Tile accumulation** with `d_que`.  
4. **WSI construction** into Zarr arrays using `wsi_da`.  
5. **Rechunk + Export** to pyramidal **OME‑TIFF**.  
6. **(Optional)** QC visualization & metadata audit.

```text
[Camera] → [DL Autofocus] → [Tile Buffer] → [Zarr Pyramid]
                               │                 │
                           [Blend/Seam]      [OME‑TIFF]
```

---

## 📁 Data & Annotations (Guidance)

- **Input:** Brightfield/fluorescence tile patches with known z‑offsets for supervised training.  
- **Labels:** Continuous focus distances (µm/mm) or equivalent defocus proxy (e.g., objective steps).  
- **Augmentations:** Contrast, mild blur/jitter, intensity scaling; preserve focus ordering.  
- **QC:** Hold‑out slide regions across staining conditions to avoid site‑specific bias.

> The archive does not include datasets. Use institution‑approved de‑identified data only.

---

## 🔐 Compliance & Ethics

- Follow institutional guidelines for handling clinical images.  
- Remove PHI/PII and respect consent policies.  
- Validate across staining protocols, scanners, magnifications, and tissue types before deployment.

---

## 🔧 Extensibility

- Swap backbones (e.g., EfficientNet, ConvNeXt) for the regressor.  
- Add sharpness priors or self‑supervised objectives.  
- Integrate learned blending or seam‑cutting for challenging tissues (fatty, folded, or sparse slides).  
- Plug in GPU‑accelerated codecs and different compression schemes for pyramids.

---

## 📚 Citations & Related Standards

- **OME‑TIFF / OME‑Zarr** (Open Microscopy Environment) standards.  
- **Dask/Zarr** for chunked, out‑of‑core arrays.  
- **ResNet** family for CNN regression backbones.

/stitching strategy are **patent‑protected**; redistribution and commercial use may require a separate agreement.  
- Trademarks belong to their respective owners.

---

### Acknowledgements

Thanks to contributors in microscopy/open‑source imaging ecosystems (OME, Zarr, Dask, PyTorch, OpenCV) and to the pathology engineers who helped validate the inline autofocus at scale.
