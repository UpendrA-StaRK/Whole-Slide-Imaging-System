# Whole-Slide Imaging System (WSI) — Deep-Learning Autofocus & Pyramidal OME-TIFF

> Python · TensorFlow/PyTorch · OpenCV · Dask/Zarr · OME-TIFF

Core components of a Whole-Slide Imaging pipeline for high-throughput digital pathology. The code covers a deep-learning autofocus regressor for inline focus estimation, a stitched WSI assembly flow, and a memory/I/O-optimized exporter that produces pyramidal OME-TIFF for common viewers and downstream analysis.

---

## 🚀 Highlights

* **Deep-Learning Autofocus (Regressor):** Learns continuous focus distance from patches, enabling **±0.002 mm** focus accuracy across multi-resolution slides.
* **Seamless WSI Capture & Stitching:** Implements an **optimal capture strategy (patented)** to minimize seams and parallax while maximizing in-focus coverage.
* **Edge-Optimized Export:** Pyramidal OME-TIFF generation with Dask/Zarr to reduce processing time and peak RAM footprint on **NVIDIA Jetson-class** devices.
* **Modular, Inline Processing:** Built for inline scanning: estimate focus → select sharp tiles → assemble → pyramidal export.
* **Rechunking for I/O Efficiency:** Chunk geometry tuned per level for fast random access in downstream viewers.

> **IP notice:** Portions of the capture/stitching logic are patent-protected. See **License & IP**.

---

## 🧭 Repository Snapshot

This archive contains a subset of the full project focused on autofocus, accumulation utilities, and the OME-TIFF pipeline.

```
Whole-Slide-Imaging-System-main/
├─ DL_model.py        # ResNet-based focus regressor + training/inference helpers
├─ img_acc.py         # Lightweight image/patch accumulator (double-ended queue)
├─ packagep2.py       # Dask/Zarr-backed WSI data adapter and exporter
└─ packages1.py       # Multiprocessing collection, registration, OME-TIFF writer
```

### Key Modules & Classes

* **`ResNetFocusRegressor` (`DL_model.py`)**
  CNN regressor (ResNet backbone) with continuous focus prediction. Includes dataset/transform scaffolding (`InlineFocusDataset`), a training loop (`train_inline_focus_model`), and inference (`predict_focus_distance`).

* **`d_que` (`img_acc.py`)**
  Minimal double-ended queue for patch/window accumulation with `append`, `popleft`, `trim`, and random access.

* **`wsi_da` (`packagep2.py`)**
  Dask/Zarr data abstraction for WSI pyramids. Provides `rechucking`, `save_as_zarr`, and `zarr_to_ome_tiff` to move from intermediate arrays → Zarr → OME-TIFF.

* **`ProcessCollection` (`packages1.py`)**
  Multiprocessing-friendly collector for tile batches, with raster/level registration and OME-TIFF emission (`save_processed_images`). Includes `zarr_process(queue, zarr_path, ome_tiff_path)` for worker execution.

> Names and signatures reflect the code in this archive; auxiliary pieces (device control, stitching heuristics, etc.) exist in the private repository.

---

## 🔬 Autofocus Model (Inline)

**Objective:** Predict best-focus offsets from raw/near-raw image patches to proactively drive the stage, reducing z-stacking and capture time.

**Approach:**

* Supervised regression with a ResNet backbone (`ResNetFocusRegressor`).
* Data pipeline via `InlineFocusDataset` (torchvision transforms).
* Learning-rate scheduling with `ReduceLROnPlateau`.
* Exportable inference path `predict_focus_distance()` for inline use.

**Why regression?**
Continuous offsets preserve ordering information and enable sub-step correction—critical for **±0.002 mm** accuracy.

---

## 🧵 Stitching & Capture Strategy (Patented)

The capture algorithm plans a path over the specimen with overlap tuned to field curvature and depth of field. Inline focus scores trigger selection or re-capture of tiles that miss the sharpness threshold. Neighbor-aware blending is applied during assembly to avoid visible seams.

> The archive includes only the interfaces required by the exporter and accumulator for IP-protected components.

---

## 🗂️ Pyramidal OME-TIFF Export

* **Intermediate Format:** Zarr arrays (chunked, compressed) for scalable out-of-core processing.
* **Rechunking:** `wsi_da.rechucking()` optimizes chunk geometry per pyramid level for viewer access patterns.
* **Emission:** `wsi_da.zarr_to_ome_tiff()` writes standards-compliant OME-TIFF with multi-resolution pyramids.
* **Multiprocessing:** `ProcessCollection` streams tiles to persistent storage without exceeding Jetson memory budgets.

**Why OME-TIFF?**
Interoperable metadata and multi-resolution pyramids enable plug-and-play use with pathology viewers and analysis pipelines.

---

## 📈 Performance Notes

* **Focus Accuracy:** ±0.002 mm across multi-resolution slides (validated on representative datasets).
* **Throughput:** Pyramidal export via Dask/Zarr reduces wall-clock time relative to monolithic in-memory assembly.
* **Memory Footprint:** Peak RAM reduced to meet **NVIDIA Jetson** constraints by streaming tiles and avoiding large intermediates.

> Exact results depend on optics, camera, tile size, pyramid depth, compression, and overlap ratio.

---

## 🧩 Typical Pipeline

1. Tile acquisition via stage control and camera SDK
2. Inline focus prediction with `ResNetFocusRegressor` → select/correct
3. Patch/Tile accumulation with `d_que`
4. WSI construction into Zarr arrays with `wsi_da`
5. Rechunk → export to pyramidal **OME-TIFF**
6. (Optional) QC visualization & metadata audit

```text
[Camera] → [DL Autofocus] → [Tile Buffer] → [Zarr Pyramid]
                               │                 │
                           [Blend/Seam]      [OME-TIFF]
```

---

## 🔧 Extensibility

* Swap backbones (e.g., EfficientNet, ConvNeXt) for the regressor.
* Add sharpness priors or self-supervised objectives.
* Integrate learned blending or seam-cutting for challenging tissues.
* Use GPU-accelerated codecs or alternate compression schemes for pyramids.

---

## 📚 References & Standards

* **OME-TIFF / OME-Zarr** (Open Microscopy Environment) standards
* **Dask/Zarr** for chunked, out-of-core arrays
* **ResNet** family for CNN regression backbones

---

### Acknowledgements

Thanks to contributors in microscopy and open-source imaging ecosystems (OME, Zarr, Dask, PyTorch, OpenCV) and to the pathology engineering teams involved in validating the inline autofocus at scale.
