#!/usr/bin/env sh

# sudo ./build/tools/ristretto quantize \
# 	--model=models/VGGNet/VOC0712/SSD_300x300_ft/train_val.prototxt \
# 	--weights=models/VGGNet/VOC0712/SSD_300x300_ft/VGG_VOC0712_SSD_300x300_ft_iter_120000.caffemodel \
# 	--model_quantized=models/VGGNet/VOC0712/SSD_300x300_ft/ssd_quantized.prototxt \
# 	--trimming_mode=dynamic_fixed_point -gpu 0,1,2,3 --iterations=25 \
# 	--error_margin=3 --detection

sudo ./build/tools/ristretto quantize \
	--train_model=models/VGGNet/VOC0712/SSD_300x300_ft/train.prototxt \
	--test_model=models/VGGNet/VOC0712/SSD_300x300_ft/test.prototxt \
	--weights=models/VGGNet/VOC0712/SSD_300x300_ft/VGG_VOC0712_SSD_300x300_ft_iter_120000.caffemodel \
	--model_quantized=examples/MobileNet-SSD/MSSD_quantized.prototxt \
	--trimming_mode=dynamic_fixed_point --gpu="0,1,2,3" --iterations=10 \
	--error_margin=3 --detection
