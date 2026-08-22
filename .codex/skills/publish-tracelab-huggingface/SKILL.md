---
name: publish-tracelab-huggingface
description: Prepare, publish, refresh, or validate the TraceLab public dataset on Hugging Face under UW-SyFI/TraceLab. Use for Hugging Face releases, Dataset Card updates, Viewer/Data Studio failures, version tags, load_dataset compatibility, or mirroring a GitHub TraceLab release while preserving exact JSONL and DuckDB assets.
---

# Publish TraceLab to Hugging Face

Publish exact GitHub release assets plus Viewer-friendly relational Parquet. Treat the release
JSONL and DuckDB as provenance authorities; never rewrite them to satisfy Hugging Face.

## Invariants

- Resolve the exact repo, release tag, JSONL, DuckDB, hashes, and row counts first.
- Keep `syfi_coding_trace.jsonl.gz` and `syfi_coding_trace.duckdb` byte-identical to GitHub.
- Derive `rounds`, `tool_calls`, and `timing_events` Parquet only from that DuckDB.
- Make `rounds` the default config; join child configs through `round_pk`.
- Stage under `$TMPDIR`; preserve unrelated dirty files.
- Use `uv run python`; never system Python or pip.
- Authenticate with explicit `HF_TOKEN`; never overwrite a shared `HF_HOME` login.
- Create/move the version tag only after the final Viewer-ready commit passes.

## 1. Audit

```bash
gh release view VERSION --repo uw-syfi/TraceLab --json tagName,assets,body,url
sha256sum trace/syfi_coding_trace.jsonl.gz trace/syfi_coding_trace.duckdb
gzip -t trace/syfi_coding_trace.jsonl.gz
git status --short --branch
```

Confirm `HfApi().whoami()` is the intended account with write/admin access to `UW-SyFI`. Set a
task-local `HF_HOME` under `$TMPDIR` so Xet logs do not use the shared cache; `HF_TOKEN` supplies auth.

## 2. Build staging

```bash
release_staging_directory="$(mktemp -d "$TMPDIR/tracelab-hf.XXXXXX")"
uv run python .codex/skills/publish-tracelab-huggingface/scripts/export_parquet.py \
  --database trace/syfi_coding_trace.duckdb \
  --output-directory "$release_staging_directory/data/VERSION"
```

The exporter converts `rounds.turn_id` UUID to string because `datasets` rejects Arrow UUID, and
casts `tool_calls.command_exit_code` HUGEINT to int64.

Copy exact JSONL/DuckDB assets into versioned auxiliary paths. Copy `LICENSE-DATASET.md` and
`NOTICE`; create `SHA256SUMS` for all data files.

The Dataset Card must configure only stable Parquet:

```yaml
configs:
  - config_name: default
    default: true
    data_files: [{split: train, path: data/VERSION/rounds/train.parquet}]
  - config_name: tool_calls
    data_files: [{split: train, path: data/VERSION/tool_calls/train.parquet}]
  - config_name: timing_events
    data_files: [{split: train, path: data/VERSION/timing_events/train.parquet}]
```

Keep nested JSONL outside `configs`: optional provider fields such as `turn_id` make the HF JSON
builder freeze an incomplete early schema and later fail with `column names don't match`.

Document counts, schemas, sanitization, responsible use, CC BY 4.0, hashes, GitHub release, project
site, `load_dataset()` examples, arXiv URL, and issue #22's conservative replay guidance. Clearly
label Parquet as derived and JSONL/DuckDB as byte-identical originals.

## 3. Validate locally

```bash
uv run python .codex/skills/publish-tracelab-huggingface/scripts/validate_local_release.py \
  --database trace/syfi_coding_trace.duckdb \
  --staging-directory "$release_staging_directory" \
  --version VERSION
```

Load all three configs locally with `uv run --with datasets python` before uploading.

## 4. Upload

Create `UW-SyFI/TraceLab` private, upload staging with `hf upload --repo-type dataset`, then read
back `repo_info(..., files_metadata=True)` and compare every LFS SHA-256. If public publication was
requested and checks pass, set `private=False`.

Private Viewer requires PRO/Enterprise; a free org returns 501. In that case validate locally first,
publish, then immediately validate the public repo.

## 5. Validate Hub and tag

Run the Hub validator in tmux because first indexing/full loading can take minutes:

```bash
TMUX_TMPDIR="$TMPDIR" tmux new-session -d -s tracelab-hf-validate -c "$(pwd)" \
  "uv run --with datasets python \
  .codex/skills/publish-tracelab-huggingface/scripts/validate_hub_release.py \
  --repo-id UW-SyFI/TraceLab --load-dataset \
  > '$TMPDIR/tracelab-hf-validate.log' 2>&1"
```

Require Viewer preview/search/filter, three Parquet configs with no pending/failed jobs, exact Hub
row counts, and exact `load_dataset()` counts. `statistics=false` alone is non-blocking when Viewer
works; HF can fail its histogram on constant-length pseudonymous `session_id` values.

Only then create the version tag at the exact final commit and verify the tag target.

## Failure routing

- `column names don't match`: JSONL is configured; switch to stable Parquet.
- `extension<arrow.uuid>`: cast `turn_id` to string.
- `command_exit_code` becomes float: cast HUGEINT to BIGINT.
- private Viewer 501: locally validate, publish, then check public Viewer.
- Xet shared-cache permission error: use task-local `HF_HOME` plus `HF_TOKEN`.
- `response is not ready yet`: poll; pending is not failure.
- failed config: inspect the Dataset page embedded traceback before changing data.

## Scripts

- `scripts/export_parquet.py`: export compatible relational Parquet.
- `scripts/validate_local_release.py`: verify checksums, gzip, counts, and Parquet types.
- `scripts/validate_hub_release.py`: poll Viewer/Parquet/size and optionally run `load_dataset`.
