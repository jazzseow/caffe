import numpy as np
import caffe
import cv2
import time

caffe.set_mode_gpu()

def compareList(list1, list2):
	if len(list1) != len(list2):
		print 'differenc length'
		return False
	for i in range(len(list1)):
		if list1[i] != list2[i]:
			# print i, list1[i], list2[i]
			return False
	return True

def checkSiam(src1, src2, net):
	path1 = '/home/ctg-sa/jazz_ws/caffe/examples/pedestrian_siamese/' + str(src1) + '.jpg'
	path2 = '/home/ctg-sa/jazz_ws/caffe/examples/pedestrian_siamese/' + str(src2) + '.jpg'

	img1 = cv2.imread(path1)
	img2 = cv2.imread(path2)

	img1 = cv2.resize(img1, (127, 127))
	img2 = cv2.resize(img2, (127, 127))

	img1 = img1.transpose((2, 0, 1))
	img2 = img2.transpose((2, 0, 1))

	data_img = np.vstack((img1, img2))
	data_img = data_img.astype(np.float32)

	# data_img = data_img * 0.00390625
	# cv2.imshow("a", data_img[1])
	# cv2.waitKey(0)
	# print data_img.shape

	net.blobs['input_data'].data[...] = data_img
	# net.blobs['sim'].data[...] = 1
	out = net.forward()

	# print src1, src2, out['loss'][1]
	# a1 = np.copy(net.blobs['conv5_flat'].data[0])
	# b1 = np.copy(net.blobs['conv5_p_flat'].data[0])
	# print a1.shape
	# print np.array_equal(a1,b1)
	# print np.sqrt(np.sum(np.square(a1 - b1))/2)
	print src1, src2, out['loss']
	# return out['pred']



def checkSiamSingle(srcs, net):
	data_img = []
	first = True
	for src in srcs:
		path = '/home/ctg-sa/jazz_ws/caffe/examples/pedestrian_siamese/' + str(src) + '.jpg'

		img1 = cv2.imread(path)
	# cv2.imshow("a", img1)
	# cv2.waitKey(0)
		img1 = cv2.resize(img1, (127, 127))
		img1 = img1.transpose((2, 0, 1))

		data_img.append(img1.astype(np.float32))

	data_img = np.asarray(data_img)
	data_img = data_img * 0.00390625

	print data_img.shape

	# print data_img.shape
	net.blobs['data'].reshape(*data_img.shape)
	net.blobs['data'].data[...] = data_img
	out = net.forward()

	# print src1, out['conv5_flat']
	print net.blobs['conv5_flat'].data.shape

def calcLoss(res1, res2):
	return np.sqrt(np.sum(np.square(res1 - res2))/2)

# net1 = caffe.Net('/home/ctg-sa/jazz_ws/caffe/examples/pedestrian_siamese/siameseFC_DW/siamese_pd_deploy.prototxt',
#                 '/home/ctg-sa/jazz_ws/caffe/examples/pedestrian_siamese/siameseFC_DW/siamese_pd_deploy.caffemodel',
#                 caffe.TEST)

net2 = caffe.Net('/home/ctg-sa/jazz_ws/caffe/examples/pedestrian_siamese/siameseDW/siameseDW_single.prototxt',
				 '/home/ctg-sa/jazz_ws/caffe/examples/pedestrian_siamese/siameseDW/siameseDW_deploy.caffemodel',
				  caffe.TEST)

# net = caffe.Net('/home/ctg-sa/jazz_ws/caffe/examples/pedestrian_siamese/siameseDW/siameseDW_deploy.prototxt',
#                 '/home/ctg-sa/jazz_ws/caffe/examples/pedestrian_siamese/siameseDW/siameseDW_deploy.caffemodel',
#                 caffe.TEST)

# for k, v in net.params.items():
# 	print k," ",v[0].data.shape
#
# print ' '
# for k, v in net.blobs.items():
# 	print k," ",v.data.shape
#
start = time.time()
# a = []
# for i in range(1,8):
# 	for j in range(1,8):
# 		i,j,checkSiam(i, j, net) #240ms

# b = []
# for i in range(1,8):
checkSiamSingle([i for i in range(1,8)], net2)
#
# for i in range(6):
# 	for j in range(6):
# 		print i+1,j+1,calcLoss(b[i], b[j])

# for i in range(6):
# 	for j in range(6):
# 		print i+1, j+1, compareList(a[i], b[j])
#
# print '\n'
# for i in range(5):
# 	print np.array_equal(b[i], b[i+1])

print (time.time() - start)*1000, 'ms'
