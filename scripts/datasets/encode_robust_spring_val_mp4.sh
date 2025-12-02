#!/usr/bin/env bash

for c in data/robust_spring/*
do
  if [[ ! -d $c ]]
  then
    continue
  fi
  for d in 0035 0046
  do
    for s in 'frame_left' 'frame_right'
    do
      # -crf 18 for visually lossless quality
      # Scale to 720p to match the training and validation datasets.
      ffmpeg -i "${c}/test/$d/$s/${s}_%04d.png" -c:v libx264 -crf 18 -filter:v scale=1280:720 "data/clips/robust_spring_$(basename "$c")_${d}_${s}.mp4"
    done
  done
done
