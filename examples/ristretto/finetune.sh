#!/usr/bin/env sh

./build/tools/caffe train \
	--solver=examples/MobileNet-SSD/solver_finetune.prototxt \
	--weights=examples/MobileNet-SSD/models/mobilenet_iter_1000.caffemodel \
	--gpu="0,1,2,3"
