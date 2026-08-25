# Hybrid dataset strategy

The service uses one normalized manifest for synthetic, public, physical, and
production examples. It deliberately does not download or redistribute
third-party datasets automatically: licence, provenance, and label quality must
be reviewed before an example becomes training eligible.

## Dataset layers

| Layer | Primary use | Training eligibility |
|---|---|---|
| Repository-generated stickers | TrOCR printed-ID bootstrap and controlled degradation coverage | Verified automatically because labels are generated together with pixels |
| Public degraded-QR datasets | QR restoration and deterministic decoder benchmarks | Pending unless exact payload labels and licence are verified |
| Printed physical stickers | Held-out camera/domain-shift evaluation | Verified after capture manifest review |
| Confirmed Tap Electric scans | Domain fine-tuning and final evaluation | Verified only through a trusted confirmation workflow |

## Public sources considered

### Damaged QR and Barcode Data — Kaggle

- URL: <https://www.kaggle.com/datasets/tmuallim/damaged-qr-and-barcode-data>
- Licence displayed by the dataset: CC BY 4.0.
- Contains paired synthetic damaged and clean QR/barcode images.
- Suitable for restoration experiments and decoder benchmarking.
- Not an EV-charger sticker dataset and should not be presented as real camera data.

### QR Code Dataset V2 — Figshare

- URL: <https://figshare.com/articles/dataset/QR_Code_Dataset_V2/28424213>
- Licence: CC BY 4.0.
- Contains unreadable scanned QR codes, unreadable simulated QR codes, and
  computer-generated learning data.
- Verify the payload annotation available for each file before using exact-match
  metrics or marking examples as training eligible.

### Other sources

Hugging Face and vendor-hosted collections can be useful for detection
benchmarks, but several lack a dataset card, have restrictive non-commercial or
no-derivatives licences, or omit decoded payload labels. They are not downloaded
by this project. Record the exact version, licence, checksum, and attribution for
any additional source before importing it.

## Normalized manifest

The importer accepts JSON containing an `examples` list. Image paths are relative
to the manifest and cannot escape its directory.

```json
{
  "schema_version": "1.0",
  "dataset_name": "physical-sticker-pilot",
  "examples": [
    {
      "example_id": "capture-0001",
      "image_path": "images/capture-0001.jpg",
      "source": "physical_pilot",
      "verified": true,
      "charger_id": "NL-TAP-E00001",
      "qr_payload": "https://example.test/charge/NL-TAP-E00001",
      "sticker_id": "printed-sticker-00001",
      "session_id": "phone-a-dark-room",
      "latitude": 52.36765,
      "longitude": 4.90414,
      "split": "test",
      "degradations": ["too_dark", "perspective"],
      "severity": 0.7,
      "native_qr_success": false,
      "image_sha256": "optional-sha256-of-image-bytes"
    }
  ]
}
```

Required fields are `example_id`, `image_path`, `charger_id`, `qr_payload`,
`latitude`, and `longitude`. Generated manifests additionally contain measured
quality signals. `verified` defaults to false, ensuring unknown public labels do
not silently enter training.

## Reproducible synthetic generation

```powershell
python -m pip install -e ".[dev,notebook]"
python -m scripts.generate_synthetic_dataset `
  --chargers 500 `
  --variants-per-charger 10 `
  --blur-variants-per-charger 2 `
  --seed 42 `
  --output data/synthetic
```

This produces 5,000 images plus `data/synthetic/manifest.json`. Variant zero is a
clean baseline; two variants per charger guarantee Gaussian- and motion-blur
coverage, while later variants use one or two seeded degradations. The generator
records the exact QR decode outcome and image-quality scores after JPEG encoding.

## Import

```powershell
python -m scripts.import_dataset --manifest data/synthetic/manifest.json
```

Import is idempotent through a source/example key. It verifies optional SHA-256
checksums, decodes each file, measures quality, stores the image through the
configured object-storage adapter, creates charger and scan metadata, and adds a
reviewable label. Re-importing a manifest skips existing examples.

## Leakage prevention

The generator assigns chargers to splits before creating variants. The importer
retains the split and sticker identity. The training dataset builder groups at
the stricter charger level and rejects:

- one sticker/charger group assigned to multiple splits;
- identical image hashes assigned to multiple splits; and
- unsupported split names.

Physical captures should remain in `test`. Production evaluation should also
include a frozen temporal set containing unseen chargers and capture sessions.

## Notebook

`notebooks/hybrid_dataset_demo.ipynb` is an executed, reproducible dataset evaluation.
It regenerates 180 examples from 30 chargers, guarantees dedicated Gaussian- and
motion-blur variants for every charger, displays clean/degraded samples, measures
QR recovery by degradation, visualizes quality signals, and proves split isolation.
It does not download TrOCR or report synthetic results as production accuracy.
