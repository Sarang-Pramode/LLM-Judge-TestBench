# data/samples

Synthetic datasets used for local demos and tests. **Not** real customer
data.

## Files

- `retail_support.csv` / `.xlsx` / `.json` / `.parquet` - the same 12
  rows exported in each supported format so every loader path can be
  exercised end-to-end. Source columns are intentionally different
  from the normalized schema (e.g. `id`, `prompt`, `response`, `topic`)
  so the upload flow demonstrates column mapping.
- `full_schema_sample.csv` / `.json` - every documented normalized
  column is present and populated. Source column names match the
  normalized schema (so mapping is trivial), and the four rows
  collectively exercise all four accepted shapes for `retrieved_context`:
    1. Plain `list[str]` (classic chunked RAG).
    2. `list[dict]` with per-chunk metadata (`doc_id`, `score`, ...).
    3. A single dict (one structured document).
    4. A free-form text blob (whole doc as a string).
  Use this when exercising judge prompts against rich inputs.
- `minimal_required.csv` - the bare-minimum required columns only
  (`record_id`, `user_input`, `agent_output`, `category`). Useful for
  checking the "mapping is trivially complete" happy path.
- `malformed_missing_category.csv` - intentionally missing the required
  `category` column. Used to verify the validation error path.
- `ground_truth_eval_sample.csv` - 16 rows with full pillar SME labels,
  `retrieved_context`, `chat_history`, and **reviewer** columns
  (`reviewer_name`, `reviewer_id`) so **Reviewer analytics** works after
  upload. Use the companion preset
  `configs/mappings/ground_truth_eval_sample.yaml` (identity mapping) or
  rely on auto-suggest; after switching from another dataset, open
  **Upload** again so column mapping refreshes for the new file.

## Regenerate

```
python scripts/generate_sample_data.py
```

The companion mapping preset for `retail_support.*` lives at
`configs/mappings/retail_support.yaml`.
