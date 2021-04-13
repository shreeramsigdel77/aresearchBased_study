import cv2
import numpy as np

# input_img = "/home/pasonatech/workspace/deepNN_py/pytorch_autoencoder/cust_dc_img/resize_output/Screenshot from 2021-04-01 20-44-21.png"
# img_inference = "/home/pasonatech/workspace/deepNN_py/pytorch_autoencoder/cust_dc_img/inference/8.png"


input_img = "/home/pasonatech/workspace/deepNN_py/pytorch_autoencoder/cust_dc_img/test_data/in.png"
img_inference = "/home/pasonatech/workspace/deepNN_py/pytorch_autoencoder/cust_dc_img/test_data/output.png"


img1 = cv2.imread(input_img)
img2 = cv2.imread(img_inference)

diff = cv2.subtract(img1, img2)

#invert color
# diff = cv2.bitwise_not(diff) # OR


img1_g = cv2.imread(input_img,0)
img2_g = cv2.imread(img_inference,0)

diff_g = cv2.subtract(img1_g, img2_g)



cv2.imshow("Preview", img2_g)
cv2.waitKey(2000)

result = not np.any(diff) #returns false when diff is all zero and with not it will inverse to true

cv2.imwrite("difference1.png",diff)
cv2.imshow("Preview1", diff)
cv2.waitKey(2000)

print(diff)
print(type(diff))
cv2.imwrite("difference_g1.png",diff_g)
cv2.imshow("Preview2", diff_g)
cv2.waitKey(2000)