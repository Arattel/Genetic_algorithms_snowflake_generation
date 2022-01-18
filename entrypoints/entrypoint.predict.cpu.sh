#!/bin/bash
set -e

BASE_MODEL_NAME=$1
WEIGHTS_FILE=$2
IMAGE_SOURCE=$3

# predict
python -m evaluater.evaluate_function \
--image-source $IMAGE_SOURCE
