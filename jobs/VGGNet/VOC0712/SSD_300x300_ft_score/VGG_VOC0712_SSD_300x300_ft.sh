cd /home/jazzseow/ssd
./build/tools/caffe train \
--solver="models/VGGNet/VOC0712/SSD_300x300_ft_score/solver.prototxt" \
--weights="models/VGGNet/VOC0712/SSD_300x300_ft/VGG_VOC0712_SSD_300x300_ft_iter_120000.caffemodel" \
--gpu 0,1,2,3 2>&1 | tee jobs/VGGNet/VOC0712/SSD_300x300_ft_score/VGG_VOC0712_SSD_300x300_ft_test120000.log
