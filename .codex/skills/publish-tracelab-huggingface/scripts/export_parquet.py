#!/usr/bin/env python3
"""Export HF-compatible TraceLab Parquet tables from a release DuckDB."""

import argparse
import hashlib
import json
from pathlib import Path

import duckdb


TABLE_QUERIES = {
    "rounds": "SELECT * REPLACE (CAST(turn_id AS VARCHAR) AS turn_id) FROM rounds",
    "tool_calls": "SELECT * REPLACE (CAST(command_exit_code AS BIGINT) AS command_exit_code) FROM tool_calls",
    "timing_events": "SELECT * FROM timing_events",
}


def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path)
    arguments = parser.parse_args()

    database_path = arguments.database.resolve()
    if not database_path.is_file():
        parser.error(f"database does not exist: {database_path}")
    output_directory = arguments.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(database_path), read_only=True)
    # A manifest may be uploaded with the release. Keep host-specific directory
    # names out of it while retaining the source artifact identity.
    manifest = {"source_database": database_path.name, "tables": {}}

    for table_name, select_query in TABLE_QUERIES.items():
        table_directory = output_directory / table_name
        table_directory.mkdir(parents=True, exist_ok=True)
        output_path = table_directory / "train.parquet"
        escaped_output_path = str(output_path).replace("'", "''")
        connection.execute(
            f"COPY ({select_query}) TO '{escaped_output_path}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
        )
        source_count = connection.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]
        parquet_count = connection.execute(
            "SELECT count(*) FROM read_parquet(?)", [str(output_path)]
        ).fetchone()[0]
        if source_count != parquet_count:
            raise RuntimeError(f"{table_name}: {source_count} != {parquet_count}")
        manifest["tables"][table_name] = {
            "path": str(output_path.relative_to(output_directory)),
            "rows": parquet_count,
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        }

    manifest_text = json.dumps(manifest, indent=2, sort_keys=True)
    if arguments.manifest_output:
        arguments.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.manifest_output.write_text(manifest_text + "\n", encoding="utf-8")
    print(manifest_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
