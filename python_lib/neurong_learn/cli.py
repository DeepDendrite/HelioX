from __future__ import annotations

import argparse
import json
import sys

from .bundle import ExportBundle


def _cmd_bundle_info(args: argparse.Namespace) -> int:
    bundle = ExportBundle(export_dir=str(args.export_dir))
    bundle.require_exists()
    meta = {
        "export_dir": bundle.export_dir,
        "neurong_config": bundle.read_neurong_config(),
        "neurong_metadata": bundle.read_neurong_metadata(),
        "neurong_bundle": bundle.read_bundle_meta(),
    }
    sys.stdout.write(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m neurong_learn", add_help=True)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_info = sub.add_parser("bundle-info", help="print bundle JSON metadata (no NEURON required)")
    p_info.add_argument("--export-dir", required=True, help="bundle/export directory")
    p_info.set_defaults(_fn=_cmd_bundle_info)

    args = parser.parse_args(argv)
    return int(args._fn(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

