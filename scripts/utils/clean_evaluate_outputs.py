#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
import shutil


def main():
    parser = ArgumentParser()
    parser.add_argument("base_dirpaths", nargs="+", help="the base directory")
    args = parser.parse_args()

    runs_to_remove = []
    for base_dirpath in args.base_dirpaths:
        base_dirpath = Path(base_dirpath)
        if not base_dirpath.is_dir():
            continue
        for config_filepath in base_dirpath.rglob("_config.yml"):
            run_dirpath = config_filepath.parent.resolve()
            all_run_dirpaths = sorted(run_dirpath.parent.iterdir())
            if all_run_dirpaths[-1].resolve() != run_dirpath:
                runs_to_remove.append(run_dirpath)

    if len(runs_to_remove) > 0:
        print("You are about to remove the following run directories:")
        for run_dirpath in runs_to_remove:
            print(run_dirpath)
        if input('Type "y" to proceed: ') == "y":
            for run_dirpath in runs_to_remove:
                shutil.rmtree(run_dirpath)
            print("Done.")


if __name__ == "__main__":
    main()
