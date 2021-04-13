# from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# def img_hist(img_path):
#     img = Image.open(img_path) #rgb
#     r, g, b = img.split()
#     len(r.histogram())
#     ### 256 ###
    
#     r.histogram()
def grayimg_hist(img_path):
    file_name = os.path.split(img_path)[1]
    img = cv2.imread(img_path,0) #loads grayimg

    # create a mask
    # mask = np.zeros(img.shape[:2], np.uint8)
    # mask[100:300, 100:400] = 255
    # masked_img = cv2.bitwise_and(img,img,mask = mask)

    # Calculate histogram with mask and without mask
    # Check third argument for mask
    hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
    # hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])

    # plt.subplot(221), plt.imshow(img, 'gray')
    # plt.subplot(222), plt.imshow(mask,'gray')
    # plt.subplot(223), plt.imshow(masked_img, 'gray')
    # plt.subplot(224), 
    plt.plot(hist_full), 
    # plt.plot(hist_mask)
    plt.xlim([0,256])
    plt.title("GrayScale")
    plt.savefig(f"{file_name}_gray_hist.png")
    plt.show()



def img_hist(img_path):
    file_name = os.path.split(img_path)[1]
    img = cv2.imread(img_path) #bgr
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        print(f"{col}",histr)

        plt.plot(histr,color = col, label= col)
        plt.xlim([0,256])
        plt.title(col)
        plt.savefig(f"{file_name}_{col}_hist.png")
    plt.legend()
    plt.title("RGB")
    plt.savefig(f"{file_name}_rgb_hist.png")
    plt.show()


def img_hist_compare(img_path1,img_path2):
    file_name1 = os.path.split(img_path1)[1]
    img = cv2.imread(img_path1) #bgr
    file_name2 = os.path.split(img_path2)[1]
    img1 = cv2.imread(img_path2) #bgr
    color = ('b','g','r')
    
    for i, col in enumerate(color):
        if col == 'b':
            color2 = '#00FFFF'
        elif col == 'g':
            color2 = '#32CD32'
        elif col == 'r':
            color2 = "#800000"
        
        histr1 = cv2.calcHist([img],[i],None,[256],[0,256]) #b
        histr2 = cv2.calcHist([img1],[i],None,[256],[0,256]) #b
        
        plt.plot(histr1,color = col, label= f"{col}_in")
        plt.plot(histr2,color = color2, label= f"{col}_out")
        plt.xlim([0,256])
        plt.title(col)
        plt.legend()
        plt.savefig(f"{file_name1}_{file_name2}_{col}_hist.png")
        plt.show()
    
    # plt.title("RGB")
    # plt.savefig(f"combined_rgb_hist.png")
    # plt.show()


# input_img = "/home/pasonatech/workspace/deepNN_py/pytorch_autoencoder/cust_dc_img/resize_output/Screenshot from 2021-04-01 20-44-21.png"
# img_inference = "/home/pasonatech/workspace/deepNN_py/pytorch_autoencoder/cust_dc_img/inference/8.png"

input_img = "/home/pasonatech/workspace/deepNN_py/pytorch_autoencoder/cust_dc_img/test_data/in.png"
img_inference = "/home/pasonatech/workspace/deepNN_py/pytorch_autoencoder/cust_dc_img/test_data/output.png"
# img_hist(input_img)
# grayimg_hist(input_img)

# img_hist(img_inference)
# grayimg_hist(img_inference)


img_hist_compare(input_img,img_inference)