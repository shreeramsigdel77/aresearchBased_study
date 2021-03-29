from skimage.metrics import structural_similarity as ssim

import matplotlib.pyplot as plt

import numpy as np
import cv2


#compare two images
def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])

	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err


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
        plt.title("Augmented")
    plt.axis("off")
    # show the images
    # plt.show()
    plt.savefig(f'{title}.png')


imgA = cv2.imread("../test_images/1.jpeg")
imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
imgB = cv2.imread("../test_images/1.jpeg")
imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)

imgC = cv2.imread("../test_images/5.jpeg")
imgC = cv2.cvtColor(imgC, cv2.COLOR_BGR2RGB)
# cv2.imshow("Preview",imgA)
# cv2.waitKey(1000)
compare_images(imgA, imgA, 'Original')
for i in range (2,6):
    print(i)
    imgB = cv2.imread(f"../test_images/{i}.jpeg")
    imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)
    compare_images(imgA, imgB, f'{i}')
