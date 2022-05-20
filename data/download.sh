#!/usr/bin/env bash

yt-dlp --restrict-filenames $1 -P "data/music_video"

for filename in data/music_video/*; do

	< /dev/null ffmpeg -i "$filename" -vn -sample_rate 44100 "data/audio/$1.wav"

	< /dev/null ffmpeg -i "$filename" -an -r 25 -s 640x640 "data/video/$1.mp4"

done

rm data/music_video/*
