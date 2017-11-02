import numpy as np
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

caffe.set_mode_cpu()

# Load the original network and extract the fully connected layers' parameters.
net1 = caffe.Net('examples/MobileNet-SSD/model_quantized/MobileNetSSD_quantized_deploy.prototxt',
                 'examples/MobileNet-SSD/model_quantized/MobileNetSSD_quantized_deploy.caffemodel',
                  caffe.TEST)

# net2 = caffe.Net('lenet.prototxt',
#                 'zero.caffemodel',
#                 caffe.TEST)
# for each layer, show the output shape
ws = np.array([])
for k, v in net1.params.items():
    print k," ",v[0].data.shape
    ws = np.hstack((ws, v[0].data.flatten()))

print np.histogram(np.absolute(ws), bins=[0,1e-6,100])[0][:50]
print np.histogram(np.absolute(ws), bins=[0,1e-6,100])[1][:50]
# plt.hist(np.absolute(ws), bins=1000, label='Before quantization')
# plt.show()


for k, v in net1.params.items():
    v[0].data[ np.absolute(v[0].data) < np.histogram(np.absolute(ws),bins = [0,1e-6,100])[1][1]] = 0
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
net1.save('examples/MobileNet-SSD/MobileNetSSD_quantized_zero_deploy.caffemodel')
