"""python -m fiscalsim.data {fetch,status} [--refresh] [keys...]"""

import argparse
import sys

from .fetch import fetch, status


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m fiscalsim.data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download auto sources into data/raw")
    f.add_argument("keys", nargs="*", help="only these source keys")
    f.add_argument("--refresh", action="store_true", help="re-download files already present")
    sub.add_parser("status", help="list sources and the manual downloads still missing")
    args = ap.parse_args()
    if args.cmd == "fetch":
        return 1 if fetch(args.keys or None, refresh=args.refresh) else 0
    status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
