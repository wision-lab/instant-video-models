#!/usr/bin/env bash

for f in data/viper/val/img/*
do
  # -crf 18 for visually lossless quality
  video_id=$(basename "$f")
  ffmpeg -i "${f}/${video_id}_%05d.jpg" -c:v libx264 -crf 18 "data/clips/viper_${video_id}.mp4"
done
