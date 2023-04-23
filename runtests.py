import argparse

from benchmarks.benchmark import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()
    main(overwrite=args.overwrite, verbose=args.verbose)
