#!/usr/bin/env sh

./build/tools/caffe train \
	--solver=examples/MobileNet-SSD/solver_finetune.prototxt \
	--weights=examples/MobileNet-SSD/models/ \
	--gpu="0,1,2,3"
