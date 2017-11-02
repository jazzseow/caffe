import matplotlib
matplotlib.use('Agg')
import numpy as np
import sys,os
import cv2
import datetime
caffe_root = '/home/jazzseow/ssd/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import matplotlib.pyplot as plt


net_file= 'examples/MobileNet-SSD/model_quantized/MobileNetSSD_quantized_deploy.prototxt'
caffe_model='examples/MobileNet-SSD/model_quantized/MobileNetSSD_quantized_deploy.caffemodel'
#test_dir = "examples/MobileNet-SSD/images"
test_dir = "data/VOCdevkit/VOC2007/JPEGImages"

if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.affemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()

caffe.set_mode_gpu();
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
    return img

def postprocess(img, out):
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect(imgfile):
    a = datetime.datetime.now()

    origimg = cv2.imread(imgfile)
    img = preprocess(origimg)

    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()
    box, conf, cls = postprocess(origimg, out)

    b = datetime.datetime.now()
    c = b - a
    ms = c.seconds * 1000.0 + c.microseconds/1000.0
    # print ms,'s'
    return ms

    # for i in range(len(box)):
    #    p1 = (box[i][0], box[i][1])
    #    p2 = (box[i][2], box[i][3])
    #    cv2.rectangle(origimg, p1, p2, (0,255,0))
    #    p3 = (max(p1[0], 15), max(p1[1], 15))
    #    title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
    #    cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
    # cv2.imshow("SSD", origimg)
    #
    # k = cv2.waitKey(0) & 0xff
    #     #Exit if ESC pressed
    # if k == 27 : return False
    # return True

meanlist = []
for f in os.listdir(test_dir):
    tt = detect(test_dir + "/" + f)
    meanlist.append(tt)


# plt.hist(np.hstack((x for x in mean_per_iter)), bins='auto')
plt.hist(meanlist, bins='auto')
plt.xlabel('Time(ms)')
plt.ylabel('Number of Image')
plt.title('Mean time : %.2f' % np.mean(meanlist))
plt.savefig('log/model_quantized_time.png', dpi=256)
