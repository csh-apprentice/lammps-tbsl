#!/usr/bin/env python3
"""Run a portable paper-figure recipe without changing the manuscript."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    recipes = json.loads((HERE / "recipes.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", nargs="?", choices=recipes)
    parser.add_argument("--list", action="store_true", help="list recipes without importing plotting libraries")
    parser.add_argument("--data-root", type=Path, default=HERE / "data")
    parser.add_argument("--output-dir", type=Path, default=HERE / "artifacts")
    parser.add_argument("--input-map", type=Path, help="JSON mapping historical relative names to replacement inputs")
    parser.add_argument("--check", action="store_true", help="check and list inputs without generating figures")
    args = parser.parse_args()
    if args.list:
        for name, recipe in recipes.items():
            print(f"{name:12s} {recipe['module']} ({len(recipe['inputs'])} input paths)")
        return 0
    if args.recipe is None:
        parser.error("choose a recipe or --list")
    os.environ["TBSL_ANALYSIS_DATA_ROOT"] = str(args.data_root.resolve())
    os.environ["TBSL_ANALYSIS_OUTPUT_ROOT"] = str(args.output_dir.resolve())
    if args.input_map:
        os.environ["TBSL_ANALYSIS_INPUT_MAP"] = str(args.input_map.resolve())
    else:
        os.environ.pop("TBSL_ANALYSIS_INPUT_MAP", None)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir.resolve() / ".cache" / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(args.output_dir.resolve() / ".cache"))
    sys.path.insert(0, str(HERE.parents[2]))
    config = importlib.import_module("tools.TBSL.analysis._config")
    inputs, missing = [], []
    for relative in recipes[args.recipe]["inputs"]:
        path = Path(config.data_path(relative))
        paths = sorted(path.glob("run_1e8_*restart.lammps")) if path.is_dir() else [path]
        if not path.exists():
            missing.append(str(path))
        inputs.extend(paths)
        if args.check:
            print(f"{'OK' if path.exists() else 'MISSING'} {path}")
    if missing:
        if not args.check:
            print("Missing inputs:\n" + "\n".join(missing), file=sys.stderr)
        return 2
    if args.check:
        return 0
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for directory in (config.RESULTS_DIR, config.APPENDIX_DIR, config.PERFORMANCE_DIR, config.REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    module = importlib.import_module("tools.TBSL.analysis." + recipes[args.recipe]["module"])
    record = {
        "recipe": args.recipe, "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv, "python": platform.python_version(),
        "dependencies": {n: importlib.metadata.version(n) for n in ("numpy", "matplotlib")},
        "data_root": str(config.DATA_ROOT), "output_root": str(output),
        "input_map": config.INPUT_MAP,
        "inputs": [{"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(set(inputs))],
        "sources": {str(p.relative_to(HERE)): sha256(p) for p in sorted(HERE.rglob("*.py")) if "artifacts" not in p.parts and "data" not in p.parts},
        "status": "running",
    }
    record_path = output / (args.recipe + "_provenance.json")
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    try:
        module.main()
        record["status"] = "complete"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        record["outputs"] = [
            {"path": str(p.relative_to(output)), "sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(output.rglob("*"))
            if p.is_file() and ".cache" not in p.parts and p != record_path
        ]
        record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Wrote figures/reports and provenance to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
