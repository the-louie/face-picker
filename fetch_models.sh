#!/bin/sh
# Download the three OpenCV model-zoo models and check their hashes.
set -eu
dir=${1:-models}
sums=$(cd "$(dirname "$0")" && pwd)/models.sha256
zoo=https://github.com/opencv/opencv_zoo/raw/main/models
mkdir -p "$dir"
curl -fsSL -o "$dir/yunet.onnx"    "$zoo/face_detection_yunet/face_detection_yunet_2023mar.onnx"
curl -fsSL -o "$dir/ediffiqa.onnx" "$zoo/face_image_quality_assessment_ediffiqa/ediffiqa_tiny_jun2024.onnx"
curl -fsSL -o "$dir/sface.onnx"    "$zoo/face_recognition_sface/face_recognition_sface_2021dec.onnx"
cd "$dir" && sha256sum -c "$sums"
