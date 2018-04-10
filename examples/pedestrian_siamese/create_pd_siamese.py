import cv2
import os
from xml.etree.ElementTree import parse, Element
import matplotlib.pyplot as plt
import numpy as np
import lmdb
import caffe
from random import choice
from caffe.proto import caffe_pb2

def main():
	img_data_list = []
	for dataset in ['VOC2007', 'VOC2012']:
		img_data ={}
		for img_name in sorted(os.listdir('../../data/VOCdevkit/' + dataset + '/JPEGImages')):
			base_name = os.path.splitext(img_name)[0]
			seq = base_name.split('_')[0]
	# 		# frame = base_name.split('_')[1]
    #
			if not seq in img_data:
				img_data[seq] = []
    #
	# 		# img = cv2.imread('../../data/VOCdevkit/' + dataset + '/JPEGImages/' + img_name)
    #
			doc = parse('../../data/VOCdevkit/' + dataset + '/Annotations/' + base_name + '.xml')
			root = doc.getroot()
    #
	# 		# width = int(root.find('./size/width').text)
	# 		# height = int(root.find('./size/height').text)
    #
			objs = root.findall('object')
			id_list = []
			for obj in objs:
				person_id = obj.find('id').text
				id_list.append(person_id)
    #
				xmin = int(obj.find('./bndbox/xmin').text)
				ymin = int(obj.find('./bndbox/ymin').text)
				xmax = int(obj.find('./bndbox/xmax').text)
				ymax = int(obj.find('./bndbox/ymax').text)

				cropped = img[ymin:ymax, xmin:xmax]
				if dataset == 'VOC2012':
					cv2.imwrite('../../data/pedestrian_siamese/train/images/' + base_name + "_" + person_id + '.jpg', cropped)
				elif dataset == 'VOC2007':
					cv2.imwrite('../../data/pedestrian_siamese/test/images/' + base_name + "_" + person_id + '.jpg', cropped)
				else:
					print("Error line31")
    #
    #
			img_data[seq].append(id_list)
		img_data_list.append(img_data)

	print 'Processing test data'
	createLMDB('pd_test_lmdb', '../../data/pedestrian_siamese/test/images', img_data_list[0], 5)
	print 'Processing train data'
	createLMDB('pd_train_lmdb', '../../data/pedestrian_siamese/train/images', img_data_list[1], 5)
	readLMDB('/home/ctg-sa/jazz_ws/caffe/examples/pedestrian_siamese/pd_train_lmdb')

def readLMDB(db_path):
	env = lmdb.open(db_path,readonly=True)
	print env.stat()
	# lmdb_txn = env.begin()
	# lmdb_cursor = lmdb_txn.cursor()
	# datum = caffe_pb2.Datum()
    #
	# for key, value in lmdb_cursor:
	# 	datum.ParseFromString(value)
    #
	# 	flat_x = np.fromstring(datum.data, dtype=np.float32)
	# 	x = flat_x.reshape(datum.channels, datum.height, datum.width)
	# 	y = datum.label
	# 	print 'label: ', y
	# 	cv2.imshow('frame1',np.transpose(np.vsplit(x, 2)[0],(1,2,0)).astype(np.uint8))
	# 	cv2.imshow('frame2',np.transpose(np.vsplit(x, 2)[1],(1,2,0)).astype(np.uint8))
	# 	cv2.waitKey(0)

def createLMDB(out_path, img_path, img_data, window_size):
	'''
		channel
		height
		width
		pixel value
		label
		id
	'''
	batch_size = 1000
	env = lmdb.open(out_path, map_size=int(1e12))
	txn = env.begin(write=True)

	batch_num = 0
	data_id = [i for i in range(1000000)]
	for seq in img_data:
		print 'Processing', seq

		seq_data = img_data[seq]
		num_frame = len(seq_data)

		# for i in range(25):
		for i in range(num_frame):

			for id1 in seq_data[i]:
				for j in range(1, window_size):
					if i + j > num_frame - 1:
						break
					if id1 in seq_data[i + j]:
						'''
						images with same id
						'''
						datum = caffe.proto.caffe_pb2.Datum()
						datum.channels = 6
						datum.height = 127
						datum.width = 127

						img1 = cv2.imread(img_path + '/' + seq + '_' + str(i+1).zfill(6) + '_' + id1 + '.jpg')
						img2 = cv2.imread(img_path + '/' + seq + '_' + str(i+j+1).zfill(6) + '_' + id1  + '.jpg')

						img1 = cv2.resize(img1, (127, 127))
						img2 = cv2.resize(img2, (127, 127))

						img1 = img1.transpose((2, 0, 1))
						img2 = img2.transpose((2, 0, 1))

						data_img = np.vstack((img1, img2))
						# print data_img.shape
						data_img = data_img.astype(np.float32)
						# cv2.imshow('frame',np.transpose(np.vsplit(x, 2)[0],(1,2,0)))
						# cv2.waitKey(0)
						datum.data = data_img.tobytes()
						datum.label = 1

						chosen_id = choice(data_id)
						str_id = '{:08}'.format(chosen_id)
						txn.put(str_id, datum.SerializeToString())

						data_id.remove(chosen_id)
						batch_num += 1

						'''
						images with diff id
						'''
						try:
							id2 = choice([x for x in seq_data[i + j] if x != id1])
						except IndexError:
							continue

						datum = caffe.proto.caffe_pb2.Datum()
						datum.channels = 6
						datum.height = 127
						datum.width = 127

						img2 = cv2.imread(img_path + '/' + seq + '_' + str(i+j+1).zfill(6) + '_' + id2  + '.jpg')
						img2 = cv2.resize(img2, (127, 127))
						img2 = img2.transpose((2, 0, 1))

						data_img = np.vstack((img1, img2))
						# print data_img.shape
						data_img = data_img.astype(np.float32)

						datum.data = data_img.tobytes()
						datum.label = 0

						chosen_id = choice(data_id)
						str_id = '{:08}'.format(chosen_id)
						txn.put(str_id, datum.SerializeToString())

						data_id.remove(chosen_id)
						batch_num += 1

						if(batch_num) % batch_size == 0:
							txn.commit()
							txn = env.begin(write=True)
							print batch_num, len(data_id)

						# print 'id', data_id

	if(batch_num) % batch_size != 0:
		txn.commit()
		print 'Last Batch'
		print batch_num, len(data_id)

	print 'Done. Total of', batch_num+1, 'entries'

if __name__ == "__main__":
	main()
