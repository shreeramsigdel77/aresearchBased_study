# a single-fully connected neural layer as encoder and as decoder
from keras import callbacks
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

#tensorboard --logdir=/tmp/autoencoder

input_img = keras.Input(shape=(28,28,1))

x = layers.Conv2D(16,(3,3),activation='relu',padding='same')(input_img)
x = layers.MaxPool2D((2,2), padding='same')(x)
x = layers.Conv2D(8,(3,3), activation='relu',padding='same')(x)
x = layers.MaxPool2D((2,2), padding='same')(x)
x = layers.Conv2D(8,(3,3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2,2), padding='same')(x)


x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


#importing mnist digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(type(x_train))

x_test_temp = x_test


x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

print(x_train.shape)
print(x_test.shape)


#normalize the test_temp data

train_data = []
for count, value in enumerate(y_train):
    if value == 1:
        train_data.append(x_train[count])

x_train = np.array(train_data)

test_data = []
for count, value in enumerate(y_test):
    if value == 1:
        test_data.append(x_test[count])

x_test = np.array(test_data)


test_data7 = []
for count, value in enumerate(y_test):
    if value == 7:
        test_data7.append(x_test_temp[count])






#trainning of autoencoder for certain epoch
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])


#decode some inputs
#note take from test set

encoder = keras.Model(input_img,encoded)
encoded_imgs = encoder.predict(x_test)

#preview of encoded images
n = 10
plt.figure(figsize=(20, 8))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(encoded_imgs[i].reshape((4, 4 * 8)).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()















decoded_imgs = autoencoder.predict(x_test)

# train_loss = tf.keras.losses.mae(decoded_imgs,x_test_temp)

# plt.hist(train_loss, bins=50)
# plt.xlabel("Train loss")
# plt.ylabel("No of examples")
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
    plt.imshow(x_test[i].reshape(28, 28))
    plt.title("Original")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.title("Reconstructed")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


plt.show()


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
    s = ssim(imageA, imageB)
    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    # show first image

    ax = fig.add_subplot(1, 2, 1)

    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.title("Original")
    plt.axis("off")

    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.title("Reconstructed")
    plt.axis("off")
    # show the images
    plt.show()


imageA = x_test[0].reshape(28, 28)

print(imageA.shape)

imageB = decoded_imgs[0].reshape(28, 28)
error_value = mse(imageA, imageB)
print(error_value)

for i in range(10):
    imageA = x_test[i].reshape(28, 28)
    imageB = decoded_imgs[i].reshape(28, 28)
    compare_images(imageA, imageB, "Compare")
