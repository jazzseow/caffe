#!/usr/bin/env sh

sudo ./build/tools/ristretto quantize \
	--model=examples/MobileNet-SSD/MobileNetSSD_test.prototxt \
	--weights=examples/MobileNet-SSD/mobilenet_iter_73000.caffemodel \
	--model_quantized=examples/MobileNet-SSD/quantized.prototxt \
	--trimming_mode=dynamic_fixed_point --iterations=1000 \
	--error_margin=3
