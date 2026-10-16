# Raw-data manifests

Raw public datasets are not committed here. Add one JSON manifest per immutable
download with at least:

```json
{
  "dataset_id": "provider/dataset/version",
  "source_url": "https://example.org/file",
  "retrieved_at": "YYYY-MM-DD",
  "sha256": "...",
  "license": "...",
  "local_path": "data/raw/provider/file",
  "adapter": "adapter-name/version",
  "split": "calibration|development|validation"
}
```

Keep raw files read-only. Adapter outputs are derived artifacts and should be
regenerable from a manifest plus versioned adapter code.
