"""Validate local sealed inputs; acquisition requires --execute. No downloads."""
import argparse
import os
from pathlib import Path
import sys
from local_runtime import ROOT, STUDIES, check_study, output_path, overlap, run_child, timeout_value, verify_published


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=STUDIES)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--lock", type=Path, help="Optional exclusive local lock file")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    try:
        timeout_value(args.timeout)
        bundle = args.inputs.absolute()
        output = output_path(args.output, bundle)
        verify_published()
        check_study(args.study, bundle)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    if not args.execute:
        print("Manifest-complete sealed inputs; execution still requires recorded versions, CUDA and scientific parity gates")
        return
    lock = None
    try:
        if args.lock:
            if overlap(args.lock, ROOT) or overlap(args.lock, bundle):
                raise ValueError("Lock must be outside the release and input bundle")
            lock = os.open(args.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        env = {k: v for k, v in os.environ.items() if not k.startswith("JEPA_") and k != "PYTHONPATH"}
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
        child = "import sys; sys.path.insert(0, sys.argv[1]); from pathlib import Path; from local_runtime import dispatch; dispatch(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), float(sys.argv[5]))"
        run_child([sys.executable, "-I", "-B", "-c", child, str(Path(__file__).resolve().parent),
                   args.study, str(bundle), str(output), str(args.timeout)], args.timeout, env)
    finally:
        if lock is not None:
            os.close(lock)
            args.lock.unlink()


if __name__ == "__main__":
    main()
