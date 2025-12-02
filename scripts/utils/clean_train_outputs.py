#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
import shutil


def main():
    parser = ArgumentParser()
    parser.add_argument("base_dirpaths", nargs="+", help="a list of base directories")
    parser.add_argument(
        "-c",
        "--remove-checkpoints",
        action="store_true",
        help="remove checkpoint .pth files",
    )
    parser.add_argument(
        "-o",
        "--remove-old-best",
        action="store_true",
        help="keep only the most recent weights_best.pth file in a sequence of runs",
    )
    args = parser.parse_args()

    runs_to_remove = []
    checkpoints_to_remove = []
    best_to_remove = []
    for base_dirpath in args.base_dirpaths:
        base_dirpath = Path(base_dirpath)
        if not base_dirpath.is_dir():
            continue
        for config_filepath in base_dirpath.rglob("_config.yml"):
            run_dirpath = config_filepath.parent.resolve()
            all_run_dirpaths = sorted(run_dirpath.parent.iterdir())
            is_old = all_run_dirpaths[-1].resolve() != run_dirpath
            epoch_complete = (run_dirpath / "epoch.txt").exists()
            if is_old and (not epoch_complete):
                runs_to_remove.append(run_dirpath)
            if args.remove_checkpoints:
                checkpoints_to_remove.extend(run_dirpath.glob("*_last.pth"))
            best_filepath = (run_dirpath / "weights_best.pth").resolve()
            if args.remove_old_best and best_filepath.exists():
                all_best_filepaths = sorted(
                    run_dirpath.parent.rglob("weights_best.pth")
                )
                if len(all_best_filepaths) > 0 and (
                    all_best_filepaths[-1].resolve() != best_filepath
                ):
                    best_to_remove.append(best_filepath)

    if len(runs_to_remove) > 0:
        print("You are about to remove the following run directories:")
        for run_dirpath in runs_to_remove:
            print(run_dirpath)
        if input('Type "y" to proceed: ') == "y":
            for run_dirpath in runs_to_remove:
                shutil.rmtree(run_dirpath)
            print("Done.")

    if len(checkpoints_to_remove) > 0:
        print("You are about to remove the following checkpoint files:")
        for checkpoint_filepath in checkpoints_to_remove:
            print(checkpoint_filepath)
        if input('Type "y" to proceed: ') == "y":
            for checkpoint_filepath in checkpoints_to_remove:
                checkpoint_filepath.unlink()
            print("Done.")

    if len(best_to_remove) > 0:
        print("You are about to remove the following old best weights:")
        for best_filepath in best_to_remove:
            print(best_filepath)
        if input('Type "y" to proceed: ') == "y":
            for best_filepath in best_to_remove:
                best_filepath.unlink()
            print("Done.")


if __name__ == "__main__":
    main()
