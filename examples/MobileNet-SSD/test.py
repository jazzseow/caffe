import cv2


imgfile = 'images/000001.jpg'

img_mat = cv2.imread(imgfile)
imgcp = cv2.resize(img_mat,(3,5))

print "original \n",imgcp

print "-100 \n",imgcp.transpose((2, 0, 1))

# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('image',imgcp)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
