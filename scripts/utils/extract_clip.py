#!/usr/bin/env python3

import subprocess
from argparse import ArgumentParser
from pathlib import Path


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "input_video", help="the video file from which the clip should be extracted"
    )
    parser.add_argument(
        "start_time", type=float, help="the start time of the clip, in seconds"
    )
    parser.add_argument(
        "duration", type=float, help="the end time of the clip, in seconds"
    )
    parser.add_argument(
        "-n",
        "--no-hwaccel",
        action="store_true",
        help="disable automatic hardware acceleration",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=Path("data", "clips"),
        help="the directory where the extracted clip should be saved",
    )
    args = parser.parse_args()

    output_filename = (
        f"{Path(args.input_video).stem}_{args.start_time}_{args.duration}.mp4"
    )
    ffmpeg_args = ["ffmpeg"]
    if not args.no_hwaccel:
        ffmpeg_args.extend(["-hwaccel", "auto"])
    ffmpeg_args.extend(
        [
            "-i",
            args.input_video,
            "-ss",
            str(args.start_time),
            "-t",
            str(args.duration),
            Path(args.output_dir, output_filename),
        ]
    )
    subprocess.run(ffmpeg_args, check=True)


if __name__ == "__main__":
    main()
