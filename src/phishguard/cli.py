"""One command line for the project (rebuild Phase 2).

    phishguard train     [--sample-size N | --full] [--seed 42] ...   train Layer-1 models (Kaggle pipeline)
    phishguard analyze   --url URL [--no-reinforcement]                score one URL, print JSON
    phishguard evaluate  [--with-tests]                                  measure a trained run; writes verified metrics + model card
    phishguard evaluate-live [--phishstats N] [--tranco N]              full system with live Playwright capture (needs internet)
    phishguard snapshot-live [--phishing N] [--tranco N]               capture every eval URL once and save it (needs internet)
    phishguard replay    [--split val|test|all]                         replay a saved snapshot through the full analysis (offline)
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
    "evaluate": ("phishguard.evaluation.evaluate", "Measure a trained run: every verified metric + model card"),
    "evaluate-live": ("phishguard.evaluation.live", "Full system with live page capture (needs internet; runs in CI)"),
    "snapshot-live": ("phishguard.evaluation.snapshot:main_freeze", "Capture every evaluation URL once and save it (needs internet)"),
    "replay": ("phishguard.evaluation.snapshot:main_replay", "Replay a saved live snapshot through the full analysis (offline)"),
    "deploy": ("phishguard.models.deploy", "Deploy the latest selected model bundle"),
    "serve": (None, "Start the Streamlit dashboard"),
}


def _usage() -> str:
    lines = ["usage: phishguard <command> [options]", "", "commands:"]
    lines += [f"  {k:<13} {v[1]}" for k, v in COMMANDS.items()]
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

    mod_name, _, func = COMMANDS[cmd][0].partition(":")
    module = importlib.import_module(mod_name)
    sys.argv = [f"phishguard {cmd}", *rest]
    getattr(module, func or "main")()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
