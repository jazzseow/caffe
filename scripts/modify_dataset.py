import argparse
import os
from xml.etree.ElementTree import parse, Element

CLASSES = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


def checkXML(chosen_classes):
	empty_img_2007 = []
	empty_img_2012 = []

	vocDevkit_root = args.dataPath
	annotations_2007 = vocDevkit_root + '/VOC2007/Annotations'
	annotations_2012 = vocDevkit_root + '/VOC2012/Annotations'
	for f in os.listdir(annotations_2007):
		doc = parse(annotations_2007 + '/' + f)
		root = doc.getroot()

		for obj in root.findall('object'):
			if not obj.find('name').text in chosen_classes:
				root.remove(obj)

		objs = root.findall('object')
		if not objs:
			empty_img_2007.append(root.find('filename').text.split('.')[0])

		doc.write(annotations_2007 + '/' + f)

	for f in os.listdir(annotations_2012):
		doc = parse(annotations_2012 + '/' + f)
		root = doc.getroot()

		for obj in root.findall('object'):
			if not obj.find('name').text in chosen_classes:
				root.remove(obj)

		objs = root.findall('object')
		if not objs:
			empty_img_2012.append(root.find('filename').text.split('.')[0])

		doc.write(annotations_2012 + '/' + f)

	return sorted(empty_img_2007), sorted(empty_img_2012)


def removeExtraInfo(empty_img_2007, empty_img_2012):
	file_paths_1 = ['data/VOC0712/test.txt',
					'data/VOC0712/test_name_size.txt',
					'data/VOC0712/trainval.txt']
	file_paths_2 = []
	file_paths_3 = []

	img_main_2007 = 'data/VOCdevkit/VOC2007/ImageSets/Main'
	img_main_2012 = 'data/VOCdevkit/VOC2012/ImageSets/Main'

	for f in os.listdir(img_main_2007):
		file_paths_2.append(img_main_2007 + '/' + f)

	for f in os.listdir(img_main_2012):
		file_paths_3.append(img_main_2012 + '/' + f)

	print 'Removing texts...'
	removeTxt(file_paths_1[:2], empty_img_2007)
	removeTxt(file_paths_1[2:], empty_img_2007, empty_img_2012)
	removeTxt(file_paths_2, empty_img_2007)
	removeTxt(file_paths_3, empty_img_2012)

	annotations_2007 = 'data/VOCdevkit/VOC2007/Annotations'
	annotations_2012 = 'data/VOCdevkit/VOC2012/Annotations'
	print 'Removing annotations...'
	removeFile(annotations_2007, empty_img_2007)
	removeFile(annotations_2012, empty_img_2012)

	jpeg_2007 = 'data/VOCdevkit/VOC2007/JPEGImages'
	jpeg_2012 = 'data/VOCdevkit/VOC2012/JPEGImages'
	print 'Removing images...'
	removeFile(jpeg_2007, empty_img_2007)
	removeFile(jpeg_2012, empty_img_2012)


def removeFile(rootPath, empty_img):
	for f in os.listdir(rootPath):
		if os.path.splitext(f)[0] in empty_img:
			os.remove(rootPath + '/' + f)


def removeTxt(file_paths, empty_img_1, empty_img_2 = []):
	# print file_paths
	for fpath in file_paths:
		temp_list = empty_img_1[:]
		temp_list.extend(empty_img_2[:])
		with open(fpath,'r+') as f:
			lines = f.readlines()
			f.seek(0)
			for line in lines:
				writeF = True
				# print 'temp: ', temp_list
				# print 'line: ', line
				if not temp_list:
					# print 'empty temp'
					break
				for img in temp_list:
					if empty_img_2:
						imgName = line.split('/')[2].split('.')[0]
						if img == imgName:
							temp_list.remove(img)
							writeF = False
							break
					else:
						if img in line:
							# print 'img: ', img, ' line: ', line
							temp_list.remove(img)
							writeF = False
							break

				if writeF:
					f.write(line)
			f.truncate()


def main(args):
	chosen_classes = args.classes
	chosen_classes = [CLASSES[i] for i in chosen_classes]
	print chosen_classes

	empty_img_2007, empty_img_2012 = checkXML(chosen_classes)
	# print empty_img_2007, empty_img_2012
	# c = raw_input('Press any button')
	removeExtraInfo(empty_img_2007, empty_img_2012)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Modify VOC dataset")
	parser.add_argument('-c', '--classes', nargs='+', type=int, required=True,
						help='\n'.join('{}: {} '.format(*k) for k in enumerate(CLASSES)))
	parser.add_argument('-d', '--dataPath', type=str, required=True,
						help='Path to VOCdevkit')
	args = parser.parse_args()
	main(args)
