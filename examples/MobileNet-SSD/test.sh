#!/bin/sh
#latest=snapshot/mobilenet_iter_73000.caffemodel
latest=$(ls -t snapshot/*.caffemodel | head -n 1)
if test -z $latest; then
	exit 1
fi
../../build/tools/caffe train -solver="solver_test.prototxt" \
--weights=$latest
# ../../build/tools/caffe test -model MobileNetSSD_train_test.prototxt -weights mobilenet_iter_73000.caffemodel -gpu 0 -iterations 100
