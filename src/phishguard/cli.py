"""One command line for the project (rebuild Phase 2).

    phishguard train     [--sample-size N | --full] [--seed 42] ...   train Layer-1 models (Kaggle pipeline)
    phishguard analyze   --url URL [--no-reinforcement]                score one URL, print JSON
    phishguard evaluate  [...]                                          URL-suite benchmark (full evaluation arrives in Phase 3)
    phishguard deploy    [...]                                          copy a trained run's model bundle into outputs/models
    phishguard serve     [--port 8501]                                  start the Streamlit dashboard

Each subcommand forwards its remaining arguments to the module that implements it, so
`phishguard train --help` shows the full option list.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

COMMANDS = {
    "train": ("phishguard.pipelines.kaggle", "Train Layer-1 models on the Kaggle data"),
    "analyze": ("phishguard.app.dashboard", "Score one URL and print the analysis JSON"),
    "evaluate": ("phishguard.evaluation.url_suites", "Benchmark the curated URL suites"),
    "deploy": ("phishguard.models.deploy", "Deploy the latest selected model bundle"),
    "serve": (None, "Start the Streamlit dashboard"),
}


def _usage() -> str:
    lines = ["usage: phishguard <command> [options]", "", "commands:"]
    lines += [f"  {k:<9} {v[1]}" for k, v in COMMANDS.items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(_usage())
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"unknown command: {cmd}\n\n{_usage()}", file=sys.stderr)
        return 2
    if cmd == "serve":
        port = "8501"
        if "--port" in rest:
            port = rest[rest.index("--port") + 1]
        frontend = Path(__file__).resolve().parent / "app" / "frontend.py"
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(frontend),
                                "--server.address=0.0.0.0", f"--server.port={port}"])
    import importlib

    module = importlib.import_module(COMMANDS[cmd][0])
    sys.argv = [f"phishguard {cmd}", *rest]
    module.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
