#image channel operation

from PIL import Image, ImageChops

def compare(imageA, imageB):
    imageA = Image.open(imageA)
    imageA.show("A")
    imageB = Image.open(imageB)
    imageB.show("B")
    diff = ImageChops.difference(imageB,imageA)
    print(diff.getbbox())
    if diff.getbbox():
        diff.show("D")


img1 = "/home/pasonatech/workspace/deepNN_py/pytorch_autoencoder/cust_dc_img/resize_output/Screenshot from 2021-04-01 20-44-21.png"
img2 = "/home/pasonatech/workspace/deepNN_py/pytorch_autoencoder/cust_dc_img/inference/2.png"


compare(img1,img2)