#!/usr/bin/env bash
# the whole codec from scratch, then comma's score. WSL, GPU, about 10 hours
#   bash run_all.sh 2>&1 | tee out/run_all.log
#   TEST=1 bash run_all.sh      a few epochs of everything (~30 min), to check it all runs before the long one
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
mkdir -p out

if [ "${TEST:-0}" = "1" ]; then
  SEG=3; POSE=2; MAPS=2; MAPS_ROUND=1; CLEAN=1; CLEAN_ROUND=1; MORE=1; MORE_ROUND=1; HEAD=2; CORRECTION=2; CORRECTION_ROUND=1
else
  SEG=600; POSE=500; MAPS=600; MAPS_ROUND=100; CLEAN=60; CLEAN_ROUND=80; MORE=200; MORE_ROUND=120; HEAD=600; CORRECTION=4800; CORRECTION_ROUND=200
fi

[ -f out/targets.npz ] || python ask_judges.py                                  # the judges' answers
python train_frame_model.py seg $SEG                                            # SegNet stage
python train_frame_model.py pose $POSE                                          # PoseNet stage
python train_map_encoder.py epochs=$MAPS rounding_epochs=$MAPS_ROUND            # map guesser, judges' maps
python clean_maps.py $CLEAN                                                     # clean the maps
python train_map_encoder.py maps=clean_maps.npz continue=1 epochs=0 rounding_epochs=$CLEAN_ROUND             # map guesser on the clean maps
python train_map_encoder.py maps=clean_maps.npz continue=1 epochs=$MORE rounding_epochs=$MORE_ROUND lr=1e-3
python pose_correction.py head $HEAD                                            # retrain frame 1 on the shipped model
python pose_correction.py correction $CORRECTION $CORRECTION_ROUND              # frame 1's pose correction
CUDA_VISIBLE_DEVICES= python pack.py                                            # archive.zip (CPU)

# comma's score, on the CPU like their grader
sed -i 's/\r$//' inflate.sh ../../evaluate.sh
rm -rf archive inflated
cd ../..
CUDA_VISIBLE_DEVICES= bash evaluate.sh --submission-dir submissions/judge_map_codec --device cpu
echo RUN_ALL_DONE
