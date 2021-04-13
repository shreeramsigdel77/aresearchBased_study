from skimage.metrics import structural_similarity as ssim

import imutils
import cv2



def load_img(path):
    col_img = cv2.imread(path)
    gray_img = cv2.cvtColor(col_img, cv2.COLOR_BGR2GRAY)
    return col_img,gray_img


# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned


def structural_diff(gray1,gray2):
    (score,diff) = ssim(gray1,gray2,full=True)
    diff = (diff*255).astype("uint8")
    print("SSIM: {}".format(score))
    return diff


# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ

def threshold_diff(diff,imageA,imageB):
    raw_img = imageA.copy()
    ano_raw_img = imageB.copy()
    thresh = cv2.threshold(diff, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]     #Otsu's bimodal thresholding
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    #img before contours
    # raw_img = imageA
    cv2.imshow("Original0", raw_img)
    for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # show the output images
    cv2.imshow("Original", imageA)
    cv2.imshow("Modified", imageB)
    cv2.imshow("Diff", diff)
    cv2.imshow("Thresh", thresh)

    cv2.imwrite("diff.png",diff)
    cv2.imwrite("org_box.png",imageA)
    cv2.imwrite("anomaly.png",imageB)
    cv2.imwrite("thresh.png",thresh)
    cv2.imwrite("original.png",raw_img)
    cv2.imwrite("original_diff.png",raw_img)
    cv2.imwrite("anomaly_raw.png",ano_raw_img)
    
    
    cv2.waitKey(0)


imga = "/home/pasonatech/workspace/deepNN_py/error_call/loss_calc/10im_1_input.png"
imgb="/home/pasonatech/workspace/deepNN_py/error_call/loss_calc/0_perfect.png"
imgb="/home/pasonatech/workspace/deepNN_py/error_call/loss_calc/10im_1_anomaly.png"

# imga = "/home/pasonatech/workspace/deepNN_py/error_call/loss_calc/in_backup/10im_1.png"
# imgb="/home/pasonatech/workspace/deepNN_py/error_call/loss_calc/in_output/10im_1.png"

col_img1,img_gray1 = load_img(imga)
col_img2, img_gray2 = load_img(imgb)

#SSIM
diff = structural_diff(img_gray1,img_gray2)

#threshold diff
threshold_diff(diff,col_img1,col_img2)