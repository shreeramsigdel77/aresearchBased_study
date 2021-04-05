# a single-fully connected neural layer as encoder and as decoder
from keras import callbacks
from matplotlib.colors import Colormap

from skimage.metrics import structural_similarity as ssim
from enum import auto
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from json import detect_encoding
import keras
from keras import layers
from tensorflow.python.keras.backend import shape
from tensorflow.keras import layers, losses
from keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.preprocessing import image
import os,cv2
import numpy as np
#create training images

#to avoid gpu errors
# physical_devices = tf.config.list_logical_devices("GPU")
# print(physical_devices)
# tf.config.experimental.set_memory_growth(physical_devices[0],True)


def process_img(dir_path):
    """[converts images into numpy array to load a dataset]

    Args:
        dir_path ([type]): [Path to image folder]

    Returns:
        [numpy.ndarray]: [list of images]
    """
    img_list = []
    for images in os.listdir(dir_path):
        if '.png' in images:
            #load image
            img = load_img(os.path.join(dir_path, images))

            # print("original", type(img))

            #convert to numpy array
            img_array = img_to_array(img)

            # print("typeee:", type(img_array))

            # print("type:", img_array.dtype)
            # print("shape:", img_array.shape)
            img_list.append(img_array)
    img_list = np.asarray(img_list)

    return img_list




#tensorboard --logdir=/tmp/autoencoder

input_img = keras.Input(shape=(400,400,3))


#######################encoding#####################################
x = layers.Conv2D(16,(3,3),activation='relu',padding='same')(input_img)
x = layers.MaxPool2D((2,2), padding='same')(x)

x = layers.Conv2D(8,(3,3), activation='relu',padding='same')(x)
x = layers.MaxPool2D((2,2), padding='same')(x)

x = layers.Conv2D(8,(3,3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2,2), padding='same')(x)

####################Deconvolutional#################################
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)

x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)

x = layers.Conv2D(8, (3, 3), activation='relu', padding= 'same')(x)
x = layers.UpSampling2D((2, 2))(x)

##################################################################
decoded = layers.Conv2D(3, (3, 3), activation='tanh', padding='same')(x)

# decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# autoencoder.summary()

#importing mnist digits
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(type(x_train))


########################################################################

train_dir = "/home/pasonatech/workspace/deepNN/aresearchBased_study/fabric/data/train"
test_dir = "/home/pasonatech/workspace/deepNN/aresearchBased_study/fabric/data/val"
inference_dir = "/home/pasonatech/workspace/deepNN/aresearchBased_study/fabric/data/resize_output"

results_dir = "/home/pasonatech/workspace/deepNN/aresearchBased_study/fabric/inference_results"
batch_size = 1

x_train = process_img(train_dir)
x_test = process_img(test_dir)
x_inference = process_img(inference_dir)

print(x_inference)
print(type(x_test))
print(type(x_test))
print(type(x_inference))
#normalized
x_train = x_train.astype('float16')/ 255.
x_test = x_test.astype('float16') / 255.
x_inference = x_inference.astype('float16')/255.
# reshapping numpy array
# x_train = np.reshape(x_train, (len(x_train), 400, 400, 3))
# x_test  = np.reshape(x_test,  (len(x_test), 400, 400, 3))            

# print(x_test)
# print(x_train)
# 
# print(type(x_test[0]))
# exit()
# n = 5
# plt.figure(figsize=(20, 4))
from PIL import Image as im
# for i in range(n):
#     #Display original
#     ax = plt.subplot(2, n, i+1)
#     x_img = x_inference[i]
#     img = array_to_img(x_inference[i])
#     img = np.array(img)
#     print(x_img.shape)
#     # array = np.reshape(x_img, (400, 400,3))
#     # image = im.fromarray(array)
#     plt.imshow(img)
#     plt.title("Original")
#     # plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#     #Display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(img)
#     plt.title("Reconstructed")
#     # plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)


# plt.show()




#reshape mapping

autoencoder.fit(x_train,x_train,
                epochs=50,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(x_test,x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')],
                )
#evaluate 
# autoencoder.evaluate(x_test,x_test, batch_size=batch_size)

autoencoder.save_weights('save_model_50/')


# decode some inputs
# note take from test set

encoder = keras.Model(input_img,encoded)
encoded_imgs = encoder.predict(x_test)

#preview of encoded images
# n = 10
# plt.figure(figsize=(20, 8))
# for i in range(1, n + 1):
#     ax = plt.subplot(1, n, i)
#     plt.imshow(encoded_imgs[i].reshape((4, 4 * 8)).T)
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()



decoded_imgs = autoencoder.predict(x_test)

total_value_x_test = np.sum(x_test[0])
total_value = np.sum(decoded_imgs[0])
print("Total value test", total_value_x_test)
print("Total value", total_value)

#matplot for visualization
#no of digits to be displayed
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    #Display original
    ax = plt.subplot(2, n, i+1)
    img = array_to_img(x_test[i])
    img_org = np.array(img)
    # print(x_img.shape)
    plt.imshow(img_org)
    plt.title("Original")
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    img = array_to_img(decoded_imgs[i])
    img_re = np.array(img)
    # print(x_img.shape)
    plt.imshow(img_re)
    plt.title("Reconstructed")
   
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.savefig(os.path.join(results_dir, "validation_img.png"))
# plt.show()


#compare two images
def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])

	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err


def compare_images(imageA, imageB, title):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB, multichannel= True)
    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    # show first image

    ax = fig.add_subplot(1, 2, 1)

    plt.imshow(imageA)
    plt.title("Original")
    plt.axis("off")

    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB)
    plt.title("Reconstructed")
    plt.axis("off")
    # show the images
    plt.savefig(os.path.join(results_dir, f"{title}.png"))
    # plt.show()


for i in range(len(decoded_imgs)):
    if i < 15:
        imageA = array_to_img(x_test[i])
        imageA = np.array(imageA)

        imageB = array_to_img(decoded_imgs[i])
        imageB = np.array(imageB)

        compare_images(imageA, imageB, f"TestCompare{i}")



imageA = array_to_img(x_test[0])
imageA = np.array(imageA)

# imageA = x_test[0].reshape(400, 400,3)

print(imageA.shape)

imageB = array_to_img(decoded_imgs[0])
imageB = np.array(imageB)

error_value = mse(imageA, imageB)
print(error_value)


decoded_imgs = autoencoder.predict(x_inference)

for i in range(len(decoded_imgs)):
    imageA = array_to_img(x_inference[i])
    imageA = np.array(imageA)

    imageB = array_to_img(decoded_imgs[i])
    imageB = np.array(imageB)

    compare_images(imageA, imageB, f"InferenceCompare{i}")
