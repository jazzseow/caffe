#!/bin/sh
# if ! test -f examples/MobileNet-SSD/MobileNetSSD_train.prototxt ;then
# 	echo "error: example/MobileNetSSD_train.prototxt does not exist."
# 	echo "please use the gen_model.sh to generate your own model."
#         exit 1
# fi
mkdir -p models
build/tools/caffe train -solver="examples/MobileNet-SSD/solver_train.prototxt" \
-weights="examples/MobileNet-SSD/mobilenet_iter_73000.caffemodel" \
-gpu 0,1,2,3
