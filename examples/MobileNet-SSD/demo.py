import numpy as np
import sys,os
import cv2
caffe_root = '/home/jazz/ssd/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

# caffe_model='/home/jazz/SSD/caffe/models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel'
# net_file= '/home/jazz/SSD/caffe/models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt'
net_file= 'MobileNetSSD_deploy.prototxt'
caffe_model='MobileNetSSD_deploy.caffemodel'
test_dir = "images"

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

def detect(imgfile):
    origimg = cv2.imread(imgfile)
    img = preprocess(origimg)

    img = img.astype(np.float32)
    print img
    img = img.transpose((2, 0, 1))


    net.blobs['data'].data[...] = img
    out = net.forward()
    print type(out)

    box, conf, cls = postprocess(origimg, out)

    for i in range(len(box)):
       p1 = (box[i][0], box[i][1])
       p2 = (box[i][2], box[i][3])
       p3 = (max(p1[0], 15), max(p1[1], 15))
       title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
    #    if conf[i] >= 0.7:
       print title
       cv2.rectangle(origimg, p1, p2, (0,255,0))
       cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
    cv2.imshow("SSD", origimg)

    k = cv2.waitKey(0) & 0xff
        #Exit if ESC pressed
    if k == 27 : return False
    return True
detect('images/horse.jpg')
# for f in os.listdir(test_dir):
#     if detect(test_dir + "/" + f) == False:
#        break
