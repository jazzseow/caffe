import numpy as np
import sys,os
import cv2
import xml.etree.ElementTree as et
caffe_root = '/home/jazz/ssd/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

# caffe_model='/home/jazz/SSD/caffe/models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel'
# net_file= '/home/jazz/SSD/caffe/models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt'
net_file= 'MobileNetSSD_deploy.prototxt'
caffe_model='MobileNetSSD_deploy.caffemodel'
test_dir = "images"

images_dir='../../data/VOCdevkit/VOC2007/JPEGImages/000004.jpg'
anno_dir='../../data/VOCdevkit/VOC2007/Annotations/000004.xml'

if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.affemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()
net = caffe.Net(net_file,caffe_model,caffe.TEST)

CLASSES = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

def calIOU(boxA, boxB):
    print boxB
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def preprocess(src):
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img * 0.007843
    # print img
    # cv2.imshow("SSD", img)
    # cv2.waitKey(0)
    return img

def postprocess(img, out):
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect(imgfile, annotfile):
    origimg = cv2.imread(imgfile)
    img = preprocess(origimg)

    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))


    net.blobs['data'].data[...] = img
    out = net.forward()

    box, conf, cls = postprocess(origimg, out)

    xmldoc = et.parse(annotfile)
    objlist = xmldoc.getroot().findall('object')

    truthlist = {}

    for obj in objlist:
        det = []
        det.append(obj.find('name').text)
        for side in obj.find('bndbox'):
            det.append(side.text)
        if truthlist.has_key(det[0]):
            truthlist[det[0]].append(map(int,det[1:]))
        else:
            truthlist[det[0]] = []
            truthlist[det[0]].append(map(int,det[1:]))

        p4 = (int(truthlist[det[0]][-1][0]), int(truthlist[det[0]][-1][1]))
        p5 = (int(truthlist[det[0]][-1][2]), int(truthlist[det[0]][-1][3]))
        cv2.rectangle(origimg, p4, p5, (255,0,0))


    for i in range(len(box)):
        p1 = (box[i][0], box[i][1])
        p2 = (box[i][2], box[i][3])
        p3 = (max(p1[0], 15), max(p1[1], 15))
        title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
        #    if conf[i] >= 0.7:
        print truthlist
        iou = calIOU(box[i], truthlist[CLASSES[int(cls[i])]][0])
        print iou
        print title
        cv2.rectangle(origimg, p1, p2, (0,255,0))
        cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)


    cv2.imshow("SSD", origimg)

    k = cv2.waitKey(0) & 0xff
        #Exit if ESC pressed
    if k == 27 : return False
    return True

# f = open('../../data/VOC0712/test.txt')
# lines = f.readlines()
#
# for line in lines:
#     line = line.strip()
#     path = line.split()
#     detect('../../data/VOCdevkit/' + path[0], '../../data/VOCdevkit/' + path[1])

detect('../../data/VOCdevkit/VOC2007/JPEGImages/000002.jpg', '../../data/VOCdevkit/VOC2007/Annotations/000002.xml')
