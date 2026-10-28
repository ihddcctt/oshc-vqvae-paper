"""Verify the weight-free public payload and its corrected full-test results.

Run `python scripts/verify_release.py --require-manifest` for the final release.
Without that flag, a missing release_manifest.json is a warning: all remaining
checks still run, but package-level hashes are explicitly not verified.

Manifest contract: {"files": [{"path": "relative/posix/path", "bytes": 123,
"sha256": "64 lowercase hex digits"}, ...]}. The manifest excludes itself.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any

from summarize_results import PRIMARY_DATASETS, load_json, sha256_file, summarize_dataset


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 95_000_000
BANNED_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx"}
IGNORED_DIRECTORIES = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "cache", ".cache"}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def ignored_payload(relative: Path, *, manifest_name: str = "release_manifest.json") -> bool:
    parts = tuple(part.lower() for part in relative.parts)
    return (relative.as_posix() == manifest_name
            or any(part in IGNORED_DIRECTORIES or part.endswith('.egg-info') for part in parts)
            or parts[:2] == ("artifacts", "codebooks"))


def _payload_paths(root: Path) -> tuple[dict[str, Path], list[str]]:
    files, errors = {}, []
    for parent, directories, names in os.walk(root, followlinks=False):
        parent_path = Path(parent)
        kept = []
        for name in directories:
            path = parent_path / name
            relative = path.relative_to(root)
            if any(part.lower() in IGNORED_DIRECTORIES or part.lower().endswith('.egg-info') for part in relative.parts):
                continue
            if path.is_symlink():
                errors.append(f"Symlink directory is not a portable release payload: {relative.as_posix()}")
                continue
            kept.append(name)
        directories[:] = kept
        for name in names:
            path = parent_path / name
            relative = path.relative_to(root)
            key = relative.as_posix()
            if path.is_symlink():
                errors.append(f"Symlink file is not a portable release payload: {key}")
                continue
            if relative.parts[:2] == ("artifacts", "codebooks"):
                if any(suffix.lower() in BANNED_SUFFIXES for suffix in path.suffixes):
                    errors.append(f"Forbidden model-weight format in generated directory: {key}")
                if path.stat().st_size > MAX_FILE_BYTES:
                    errors.append(f"File exceeds {MAX_FILE_BYTES} bytes in generated directory: {key}")
            if ignored_payload(relative):
                continue
            files[key] = path
    return files, errors


def _safe_manifest_target(root: Path, entry_path: Any) -> tuple[str, Path]:
    if not isinstance(entry_path, str) or not entry_path or "\\" in entry_path:
        raise ValueError(f"Manifest path must be a non-empty relative POSIX path: {entry_path!r}")
    path = PurePosixPath(entry_path)
    if path.is_absolute() or ".." in path.parts or ":" in entry_path or path.as_posix() != entry_path:
        raise ValueError(f"Unsafe or non-canonical manifest path: {entry_path!r}")
    resolved = (root / Path(*path.parts)).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Manifest target escapes the release root: {entry_path!r}")
    return entry_path, resolved


def verify_release(root: Path, *, require_manifest: bool = False) -> dict[str, Any]:
    root = Path(root).resolve()
    errors, warnings = [], []
    if not root.is_dir():
        return {"schema": "oshc-vqvae-release-verification-v1", "ok": False,
                "errors": [f"Release root does not exist: {root}"], "warnings": []}
    files, traversal_errors = _payload_paths(root)
    errors.extend(traversal_errors)
    if not files:
        errors.append("No release payload files found")
    total_bytes = 0
    for relative, path in files.items():
        size = path.stat().st_size
        total_bytes += size
        if size > MAX_FILE_BYTES:
            errors.append(f"File exceeds {MAX_FILE_BYTES} bytes: {relative} ({size})")
        if any(suffix.lower() in BANNED_SUFFIXES for suffix in path.suffixes):
            errors.append(f"Forbidden model-weight format: {relative}")
    # NPZ is a metric/codebook data format; it is deliberately not classified as
    # a weight file by extension. Generated artifacts/codebooks are unversioned.

    manifest_path = root / "release_manifest.json"
    manifest_verified = False
    manifest_entries = 0
    if not manifest_path.is_file():
        message = "release_manifest.json is missing; package-level hashes were not verified"
        (errors if require_manifest else warnings).append(message)
    else:
        initial_errors = len(errors)
        try:
            manifest = load_json(manifest_path)
            entries = manifest.get("files")
            if not isinstance(entries, list) or not entries:
                raise ValueError("release_manifest.json files must be a non-empty list")
            listed, seen_casefold = set(), set()
            for entry in entries:
                try:
                    if not isinstance(entry, dict):
                        raise ValueError("Manifest file entries must be objects")
                    relative, path = _safe_manifest_target(root, entry.get("path"))
                    if relative == "release_manifest.json":
                        raise ValueError("The release manifest cannot hash itself")
                    if relative.casefold() in seen_casefold:
                        raise ValueError(f"Duplicate/case-colliding manifest path: {relative}")
                    seen_casefold.add(relative.casefold())
                    listed.add(relative)
                    size = entry.get("bytes")
                    expected_hash = entry.get("sha256")
                    if type(size) is not int or size < 0:
                        raise ValueError(f"Invalid declared byte count: {relative}")
                    if not isinstance(expected_hash, str) or not SHA256_PATTERN.fullmatch(expected_hash):
                        raise ValueError(f"Invalid declared SHA-256: {relative}")
                    if not path.is_file():
                        raise ValueError(f"Manifest payload is missing: {relative}")
                    if path.stat().st_size != size:
                        raise ValueError(f"Manifest byte-count mismatch: {relative}")
                    if sha256_file(path) != expected_hash:
                        raise ValueError(f"Manifest SHA-256 mismatch: {relative}")
                    manifest_entries += 1
                except (OSError, ValueError) as error:
                    errors.append(str(error))
            for omitted in sorted(set(files) - listed):
                errors.append(f"Unlisted release payload: {omitted}")
            manifest_verified = len(errors) == initial_errors
        except (OSError, ValueError, TypeError) as error:
            errors.append(str(error))

    evaluations = []
    datasets = [*PRIMARY_DATASETS]
    if (root / "results/recomputed/cifar10").exists():
        datasets.append("cifar10")
    for dataset in datasets:
        try:
            evaluations.append(summarize_dataset(root / "results/recomputed" / dataset, dataset))
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f"Final {dataset} results: {error}")
    return {
        "schema": "oshc-vqvae-release-verification-v1",
        "ok": not errors,
        "release_root": str(root),
        "payload_files": len(files),
        "payload_bytes": total_bytes,
        "maximum_file_bytes": MAX_FILE_BYTES,
        "release_manifest_verified": manifest_verified,
        "verified_manifest_entries": manifest_entries,
        "excluded_from_payload_inventory": ["release_manifest.json", ".git/", "cache directories", "editable-install *.egg-info/", "artifacts/codebooks/"],
        "evaluations": evaluations,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--require-manifest", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output-json", type=Path, help="Write exclusively to a new file; prefer a location outside the release")
    args = parser.parse_args()
    report = verify_release(args.root, require_manifest=args.require_manifest)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output_json:
        try:
            with args.output_json.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
        except OSError as error:
            print(f"Cannot write new JSON output: {error}", file=sys.stderr)
            return 1
    if args.json:
        print(encoded, end="")
    else:
        print(f"Release verification: {'PASS' if report['ok'] else 'FAIL'}")
        print(f"Payload: {report.get('payload_files', 0)} files; "
              f"manifest verified: {report.get('release_manifest_verified', False)}")
        for result in report.get("evaluations", []):
            print(f"{result['dataset']}: {result['images']} images, 10 correct / 90 ordered wrong pairs, "
                  f"{result['ssim_implementation']}, finite arrays and summary match")
        for message in report["warnings"]:
            print(f"WARNING: {message}")
        for message in report["errors"]:
            print(f"ERROR: {message}", file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
