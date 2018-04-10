import numpy as np
import caffe
import cv2

caffe.set_mode_gpu()

net = caffe.Net('/home/ctg-sa/jazz_ws/caffe/examples/siamese/mnist_siamese_deploy.prototxt',
                 '/home/ctg-sa/jazz_ws/caffe/examples/siamese/mnist_siamese_iter_50000.caffemodel',
                  caffe.TEST)


image1 = '/home/ctg-sa/jazz_ws/caffe/examples/siamese/1.png'
image2 = '/home/ctg-sa/jazz_ws/caffe/examples/siamese/2.png'
image3 = '/home/ctg-sa/jazz_ws/caffe/examples/siamese/3.png'
#
# for k, v in net.params.items():
# 	print k," ",v[0].data.shape
#
# print ' '
# for k, v in net.blobs.items():
# 	print k," ",v.data.shape

def checkSiam(src1, src2):
	path1 = '/home/ctg-sa/jazz_ws/caffe/examples/siamese/' + str(src1) + '.png'
	path2 = '/home/ctg-sa/jazz_ws/caffe/examples/siamese/' + str(src2) + '.png'

	img1 = cv2.imread(path1)
	img2 = cv2.imread(path2)

	img1 = cv2.resize(img1, (28, 28))
	img2 = cv2.resize(img2, (28, 28))

	img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

	data_img = np.stack((img1, img2))
	data_img = data_img.astype(np.float32)

	data_img = data_img * 0.00390625
	# cv2.imshow("a", data_img[1])
	# cv2.waitKey(0)
	# print data_img.shape

	net.blobs['input_data'].data[...] = data_img
	net.blobs['sim'].data[...] = [1]

	out = net.forward()


	print out['loss'].shape
	print src1, src2, out['loss']


for i in range(1,5):
	for j in range(1,5):
		checkSiam(i, j)
