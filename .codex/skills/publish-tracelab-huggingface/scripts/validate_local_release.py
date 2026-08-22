#!/usr/bin/env python3
"""Validate staged TraceLab HF files against the source DuckDB."""

import argparse
import gzip
import hashlib
from pathlib import Path

import duckdb


def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_checksums(checksum_path: Path) -> dict[str, str]:
    checksums = {}
    for line_number, line in enumerate(checksum_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"{checksum_path}:{line_number}: malformed checksum")
        checksums[fields[1].lstrip("*")] = fields[0]
    return checksums


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--staging-directory", type=Path, required=True)
    parser.add_argument("--version", default="v0.0.2")
    arguments = parser.parse_args()

    database_path = arguments.database.resolve()
    staging_directory = arguments.staging_directory.resolve()
    checksums = parse_checksums(staging_directory / "SHA256SUMS")
    for relative_path, expected_digest in checksums.items():
        actual_digest = sha256_file(staging_directory / relative_path)
        if actual_digest != expected_digest:
            raise RuntimeError(f"{relative_path}: {actual_digest} != {expected_digest}")

    compressed_jsonl_path = (
        staging_directory / "data" / arguments.version / "syfi_coding_trace.jsonl.gz"
    )
    with gzip.open(compressed_jsonl_path, "rb") as compressed_file:
        while compressed_file.read(1024 * 1024):
            pass

    connection = duckdb.connect(str(database_path), read_only=True)
    parquet_paths = {
        table_name: staging_directory
        / "data"
        / arguments.version
        / table_name
        / "train.parquet"
        for table_name in ("rounds", "tool_calls", "timing_events")
    }
    for table_name, parquet_path in parquet_paths.items():
        source_count = connection.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]
        parquet_count = connection.execute(
            "SELECT count(*) FROM read_parquet(?)", [str(parquet_path)]
        ).fetchone()[0]
        if source_count != parquet_count:
            raise RuntimeError(f"{table_name}: {source_count} != {parquet_count}")
        print(f"{table_name}: {parquet_count} rows")

    turn_id_type = connection.execute(
        "DESCRIBE SELECT turn_id FROM read_parquet(?)", [str(parquet_paths["rounds"])]
    ).fetchone()[1]
    exit_code_type = connection.execute(
        "DESCRIBE SELECT command_exit_code FROM read_parquet(?)",
        [str(parquet_paths["tool_calls"])],
    ).fetchone()[1]
    if turn_id_type != "VARCHAR" or exit_code_type != "BIGINT":
        raise RuntimeError(f"bad HF types: turn_id={turn_id_type}, exit_code={exit_code_type}")
    print(f"verified {len(checksums)} checksums; HF compatibility types are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
