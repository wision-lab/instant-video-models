#!/usr/bin/env python3

import subprocess
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    parser = ArgumentParser()
    parser.add_argument("model_outputs_video")
    parser.add_argument("combined_video")
    parser.add_argument("-t", "--top-text", default="Top text")
    parser.add_argument("-b", "--bottom-text", default="Bottom text\nline 2")
    args = parser.parse_args()
    with TemporaryDirectory() as tmp_dirpath:
        tmp_dirpath = Path(tmp_dirpath)
        unstabilized_video = str(tmp_dirpath / "unstabilized.mp4")
        stabilized_video = str(tmp_dirpath / "stabilized.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                args.model_outputs_video,
                "-filter",
                f"crop=iw:ih/2:0:0,"
                f"drawtext=fontsize=48:fontcolor=white:borderw=2:text_align=C:x=(w-text_w)/2:y=24:text='Unstabilized',",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                unstabilized_video,
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                args.model_outputs_video,
                "-filter",
                f"crop=iw:ih/2:0:oh,"
                f"drawtext=fontsize=48:fontcolor=white:borderw=2:text_align=C:x=(w-text_w)/2:y=24:text='Stabilized',",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                stabilized_video,
            ],
            check=True,
        )
        p = 180
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                unstabilized_video,
                "-i",
                stabilized_video,
                "-filter_complex",
                f"hstack=inputs=2,"
                f"pad=in_w:in_h+{p * 2}:0:{p}:color=black,"
                f"drawtext=fontsize=80:fontcolor=white:text_align=C:x=(w-text_w)/2:y=({p}-text_h)/2:text='{args.top_text}',"
                f"drawtext=fontsize=48:fontcolor=white:text_align=C:x=(w-text_w)/2:y=h-text_h-({p}-text_h)/2:text='{args.bottom_text}',",
                args.combined_video,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
