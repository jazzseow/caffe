import numpy as np
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

caffe.set_mode_cpu()

# Load the original network and extract the fully connected layers' parameters.
net1 = caffe.Net('examples/MobileNet-SSD/MobileNetSSD_trainval.prototxt',
                'examples/MobileNet-SSD/mobilenet_iter_1000.caffemodel',
                caffe.TEST)

# net2 = caffe.Net('lenet.prototxt',
#                 'zero.caffemodel',
#                 caffe.TEST)
# for each layer, show the output shape

for k, v in net1.params.items():
    print k," ",v.data


#accuracy = 0.9909
#I1026 14:38:56.076452  6765 caffe.cpp:330] loss = 0.0289774 (* 1 = 0.0289774 loss)
#
# w1 = net1.params['conv2'][0]
# b1 = net1.params['conv2'][1]
# # w2 = net2.params['conv2'][0]
# # b2 = net2.params['conv2'][1]
# # print np.histogram(np.absolute(w),bins = 1000)[0][:50]
# # print np.histogram(np.absolute(w),bins = 1000)[1][:50]
# # print np.histogram(np.absolute(w),bins = 1000)[1][1]
#
# w1.data[ w1.data < np.histogram(np.absolute(w1.data),bins = 1000)[1][1]] = 0
# # b.data[...] = 0
# # print b1.data
# # print '|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||'
# # print b2.data
# net1.save('zero.caffemodel')
