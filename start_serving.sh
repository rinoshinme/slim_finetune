#!/bin/bash

SERVING_NAME='rino_violence'
MODEL_NAME='violence07'
SOURCE_DIR='/data/rinoshinme/projects/data_center/code/serving/room_check'
TARGET_DIR='/models/violence07'

docker run -p 8501:8501 --name ${SERVING_NAME} \
    --mount type=bind,source=${SOURCE_DIR},target=${TARGET_DIR} \
    -e MODEL_NAME=${MODEL_NAME} \
    -t tensorflow/serving:1.14.0-gpu
