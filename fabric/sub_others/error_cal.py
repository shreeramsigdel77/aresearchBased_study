import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


# input_img_dir = "/home/pasonatech/workspace/deepNN_py/pytorch_autoencoder/loss_calc/in_backup"
input_img_dir = "/home/pasonatech/workspace/deepNN_py/error_call/loss_calc/down_img"

# inference_img_dir = "/home/pasonatech/workspace/deepNN_py/pytorch_autoencoder/loss_calc/output"
inference_img_dir = "/home/pasonatech/workspace/deepNN_py/error_call/loss_calc/down_output"


def load_img(img_dir):
    img_dir_list = os.listdir(img_dir)
    img_dir_list.sort()
    # print(img_dir)
    np_list =[]
    for i in img_dir_list:
        # print(i)
        # print(os.path.join(img_dir,i))
        img = cv2.imread(os.path.join(img_dir,i))
        np_list.append(img)
    return np_list

def mse2(imageA, imageB):

    Y = np.square(np.subtract(imageA,imageB))
    Y = Y.mean()
    return Y


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    print(imageA.astype("float") - imageB.astype("float"))
    err_cal = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err_cal /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err_cal


# input_img_nplist = load_img(input_img_dir)

# output_img_nplist = load_img(inference_img_dir)



def compare_images(imageA, imageB, title):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB,multichannel=True)
    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    # show first image

    ax = fig.add_subplot(1, 2, 1)

    # plt.imshow(imageA, cmap=plt.cm.gray)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.title("Original")
    plt.axis("off")

    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    # plt.imshow(imageB, cmap=plt.cm.gray)
    plt.imshow(imageB, cmap=plt.cm.gray)
    if title == 'Original':
        plt.title("Original")
    else:
        plt.title("Inferenced")
    plt.axis("off")
    # show the images
    # plt.show()
    plt.savefig(f'{title}.png')
    print("SSIM",s)




# error_combine = []
# for in_img, out_img in zip(input_img_nplist,output_img_nplist):
#     err= mse(in_img,out_img)
#     err = err/ 195075
#     # err = mse2(in_img,out_img)
#     error_combine.append(err)
#     # print(err)

# loss = sum(error_combine)/len(error_combine)

# print("final",loss)

imga = "/home/pasonatech/workspace/deepNN_py/error_call/loss_calc/0_perfect1.png"
imgb="/home/pasonatech/workspace/deepNN_py/error_call/loss_calc/0_perfect.png"
# imgb="/home/pasonatech/workspace/deepNN_py/error_call/loss_calc/0.png"
imga=cv2.imread(imga,0)
# cv2.imshow("wind",imga)
# cv2.waitKey(10000)
# print(imga)
# exit()
imgb=cv2.imread(imgb,0)
er_u =mse(imga,imgb)
print(er_u)
# print(er_u/195075)

# compare_images(imga,imgb,"title")




# Y = np.square(np.subtract(imga,imgb)).mean()
# print("MSE:", Y)

#validating list
#loss 
# 0.04062359595668333
# 0.02584878767781622
# 0.02471400265923363
# 0.03859080539536076
# 0.08325308663334616
# 0.045594727444572596
# 0.07825291442393952
# 0.022221491253364093
# 0.021382778002050493
# 0.026132363770344738
# 0.029294477188260927
# 0.042714361239266946

#final loss
# 0.03988528263701995



#test data

# 0.07285135838779956
# 0.03774949714853262
# 0.017903890747148533
# 0.01482399644367551
# 0.03049346978726131
# 0.02978987277329232
# 0.03645992406766628
# 0.032635770440856075
# 0.058506668044341924


# 0.03680160531561935