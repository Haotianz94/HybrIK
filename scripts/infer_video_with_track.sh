#!/bin/bash
SCRIPT=/raid/haotian/Projects/lift3d/external/hybrik/infer_video.py 
VIDEO_DIR=/raid/jhong/tennis-video
TRACK_DIR=/home/james/lift3d/data/tennis_track
OUT_DIR=/raid/haotian/Projects/lift3d/output/tennis_pose_hybrik

CUDA_VISIBLE_DEVICES=2 python $SCRIPT $VIDEO_DIR $TRACK_DIR $OUT_DIR --part 0 6  

#CUDA_VISIBLE_DEVICES=3 python $SCRIPT $VIDEO_DIR $TRACK_DIR $OUT_DIR --part 1 6 &

#CUDA_VISIBLE_DEVICES=4 python $SCRIPT $VIDEO_DIR $TRACK_DIR $OUT_DIR --part 2 6 &

#CUDA_VISIBLE_DEVICES=5 python $SCRIPT $VIDEO_DIR $TRACK_DIR $OUT_DIR --part 3 6 &

#CUDA_VISIBLE_DEVICES=6 python $SCRIPT $VIDEO_DIR $TRACK_DIR $OUT_DIR --part 4 6 &

#CUDA_VISIBLE_DEVICES=7 python $SCRIPT $VIDEO_DIR $TRACK_DIR $OUT_DIR --part 5 6 &

wait
