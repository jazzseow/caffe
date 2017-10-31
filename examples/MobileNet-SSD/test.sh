#!/bin/sh
#latest=snapshot/mobilenet_iter_73000.caffemodel
latest=$(ls -t snapshot/*.caffemodel | head -n 1)
if test -z $latest; then
	exit 1
fi
build/tools/caffe train -solver="examples/MobileNet-SSD/solver_test.prototxt" \
--weights="examples/MobileNet-SSD/mobilenet_iter_73000.caffemodel" \
-gpu 0,1,2,3
