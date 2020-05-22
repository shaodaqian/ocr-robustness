from skimage import data, color, io
import cv2
import numpy as np

filename='exp_results/6/cooperative/ub/6_currentBest_6.66_1.png'


image1 = io.imread(filename, as_gray=True)
m = np.amin(image1)
x = np.squeeze(image1)
print(m)
x = (x - m) / (1 - (m/255))
print(np.amax(x))
cv2.imwrite(filename, x*1.15, [cv2.IMWRITE_PNG_COMPRESSION, 0])
# cv2.imwrite('exp_results/9/cooperative/9_currentBest_3.17.png',image1*256)
print(x)
