# data/samples

Synthetic datasets used for local demos and tests. **Not** real customer
data.

## Files

- `retail_support.csv` / `.xlsx` / `.json` / `.parquet` - the same 12
  rows exported in each supported format so every loader path can be
  exercised end-to-end. Source columns are intentionally different
  from the normalized schema (e.g. `id`, `prompt`, `response`, `topic`)
  so the upload flow demonstrates column mapping.
- `minimal_required.csv` - the bare-minimum required columns only
  (`record_id`, `user_input`, `agent_output`, `category`). Useful for
  checking the "mapping is trivially complete" happy path.
- `malformed_missing_category.csv` - intentionally missing the required
  `category` column. Used to verify the validation error path.

## Regenerate

```
python scripts/generate_sample_data.py
```

The companion mapping preset for `retail_support.*` lives at
`configs/mappings/retail_support.yaml`.
