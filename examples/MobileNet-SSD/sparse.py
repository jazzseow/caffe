import numpy as np
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

caffe.set_mode_cpu()

def setZero(thres):
    net = caffe.Net('examples/MobileNet-SSD/model_quantized/MNSSD_quantized_trainval.prototxt',
                     'examples/MobileNet-SSD/models/mobilenet_quantized_iter_1000.caffemodel',
                      caffe.TEST)

    ws = np.array([])
    num = 0
    for k, v in net.params.items():
        print k," ",v[0].data.shape
        num += len(v[0].data[v[0].data == 0])
        ws = np.hstack((ws, v[0].data.flatten()))

    print num
    print len(ws[ws == 0])

    print np.histogram(np.absolute(ws), bins=[0,float(thres),100])[0][:50]
    print np.histogram(np.absolute(ws), bins=[0,float(thres),100])[1][:50]

    for k, v in net.params.items():
        v[0].data[ np.absolute(v[0].data) < np.histogram(np.absolute(ws),bins = [0,float(thres),100])[1][1]] = 0

    net.save('examples/MobileNet-SSD/MobileNetSSD_quantized_zero_deploy.caffemodel')


def checkZero():
    net = caffe.Net('examples/MobileNet-SSD/model_quantized/MNSSD_quantized_trainval.prototxt',
                     'examples/MobileNet-SSD/MobileNetSSD_quantized_zero_deploy.caffemodel',
                      caffe.TEST)

    num = 0
    for k, v in net.params.items():
        num += len(v[0].data[v[0].data == 0])

    print num


setZero('1e-5')
checkZero()
