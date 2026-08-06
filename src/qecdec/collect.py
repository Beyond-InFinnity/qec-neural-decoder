"""Config-driven MWPM threshold-curve collection via sinter.

Usage:
    python -m qecdec.collect --config experiments/phase0_surface.json

The config fully determines the run; the output artifact embeds the config,
git SHA, and per-task shot/error counts so every number is reproducible.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import sinter

from .circuits import CircuitSpec, make_circuit


def git_sha(repo_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build_tasks(config: dict) -> list[sinter.Task]:
    tasks = []
    for d in config["distances"]:
        rounds = d if config.get("rounds", "d") == "d" else int(config["rounds"])
        for p in config["ps"]:
            spec = CircuitSpec(
                code=config["code"],
                distance=d,
                rounds=rounds,
                p=p,
                noise=config.get("noise", "circuit"),
            )
            tasks.append(
                sinter.Task(circuit=make_circuit(spec), json_metadata=spec.to_metadata())
            )
    return tasks


def collect(config: dict, *, num_workers: int, print_progress: bool = True) -> list[dict]:
    stats = sinter.collect(
        num_workers=num_workers,
        tasks=build_tasks(config),
        decoders=config.get("decoders", ["pymatching"]),
        max_shots=config["max_shots"],
        max_errors=config["max_errors"],
        print_progress=print_progress,
    )
    rows = []
    for s in stats:
        rows.append(
            {
                **s.json_metadata,
                "decoder": s.decoder,
                "shots": s.shots,
                "errors": s.errors,
                "seconds": s.seconds,
                "logical_error_rate": (s.errors / s.shots) if s.shots else None,
            }
        )
    rows.sort(key=lambda r: (r["decoder"], r["distance"], r["p"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None, help="output JSON artifact path")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    repo_root = args.config.resolve().parent.parent
    rows = collect(config, num_workers=args.workers)

    out = args.out or args.config.with_name(args.config.stem + ".results.json")
    out.write_text(
        json.dumps(
            {"config": config, "git_sha": git_sha(repo_root), "results": rows}, indent=2
        )
    )
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
