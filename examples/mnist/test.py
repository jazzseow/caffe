import numpy as np
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

caffe.set_mode_cpu()

# Load the original network and extract the fully connected layers' parameters.
net1 = caffe.Net('examples/mnist/lenet_train_test.prototxt',
                'examples/mnist/lenet_iter_10000.caffemodel',
                caffe.TRAIN)

# net2 = caffe.Net('lenet.prototxt',
#                 'zero.caffemodel',
#                 caffe.TEST)
# for each layer, show the output shape

net1.forward()
for layer_name, blob in net1.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)
    print np.amax(blob.data)
print '\n'
for k, v in net1.params.items():
    print k , '\t' , v[0].data.shape , '\t' , v[1].data.shape
    print np.amax(v[0].data)
# print help(net1.blobs)
# accuracy = 0.9909
# I1029 02:48:23.057803 23411 caffe.cpp:330] loss = 0.0289774 (* 1 = 0.0289774 loss)
# I1029 02:49:03.935200 23443 caffe.cpp:330] accuracy = 0.9909
# I1029 02:49:03.935216 23443 caffe.cpp:330] loss = 0.0289769 (* 1 = 0.0289769 loss)


w1 = net1.params['pool1'][0].data
b1 = net1.params['conv2'][1].data

print '\n'
print w1.shape
print '\n'
print b1.shape
# w2 = net2.params['conv2'][0].data
# b2 = net2.params['conv2'][1].data

# print np.histogram(np.absolute(w1),bins = 1000)[0][:50]
# print np.histogram(np.absolute(w1),bins = 1000)[1][:50]
#
# print np.histogram(np.absolute(w2),bins = np.histogram(np.absolute(w1),bins = 1000)[1])[0][:50]
# print np.histogram(np.absolute(w2),bins = np.histogram(np.absolute(w1),bins = 1000)[1])[1][:50]
w1[ np.absolute(w1) < np.histogram(np.absolute(w1),bins = 1000)[1][250]] = 0
# b.data[...] = 0
# print b1.data
# print '|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||'
# print b2.data
net1.save('zero.caffemodel')
