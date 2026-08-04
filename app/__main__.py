
from __future__ import annotations

import argparse
import os
import uvicorn

def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m app", description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="restart on source changes")
    parser.add_argument("--state-dir", default=None,
                        help="where measurements are cached (default: .quality_app)")
    parser.add_argument("--dataset", action="append", default=[],
                        help="register a dataset folder at startup; repeatable")
    args = parser.parse_args()

    if args.state_dir:
        os.environ["QUALITY_STATE_DIR"] = args.state_dir
    if args.dataset:
        os.environ["QUALITY_DATASETS"] = os.pathsep.join(args.dataset)

    print(f"episode quality app on http://{args.host}:{args.port}")
    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
