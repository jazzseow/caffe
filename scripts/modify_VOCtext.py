import os
import sys
# from xml.etree.ElementTree import parse, Element

voc2007_dir = 'data/VOCdevkit/VOC2007'
voc2012_dir = 'data/VOCdevkit/VOC2012'

voc2012_trainval = []
voc2007_test = []

for imgF in os.listdir(voc2012_dir + '/JPEGImages'):
	img_name = os.path.splitext(imgF)[0]
	voc2012_trainval.append(img_name)

voc2007_imgs = os.listdir(voc2007_dir + '/JPEGImages')[:]
for imgF in voc2007_imgs:
	img_name = os.path.splitext(imgF)[0]
	voc2007_test.append(img_name)

# imgNum_trainval2012 = len(voc2012_trainval)
# voc2012_train = voc2012_trainval[:imgNum_trainval2012/2]
# voc2012_val= voc2012_trainval[imgNum_trainval2012/2:]
#
# with open(voc2012_dir + '/ImageSets/Main/train.txt', 'w') as textF:
# 	for img in sorted(voc2012_train):
# 		textF.write(img + '\n')
#
# with open(voc2012_dir + '/ImageSets/Main/val.txt', 'w') as textF:
# 	for img in sorted(voc2012_val):
# 		textF.write(img + '\n')

with open(voc2012_dir + '/ImageSets/Main/trainval.txt', 'w') as textF:
	for img in sorted(voc2012_trainval):
		textF.write(img + '\n')

with open(voc2007_dir + '/ImageSets/Main/test.txt', 'w') as textF:
	for img in sorted(voc2007_test):
		textF.write(img + '\n')
