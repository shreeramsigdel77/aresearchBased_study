# a single-fully connected neural layer as encoder and as decoder
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

#size of encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5 assuming the input is 784 floats

#input image
input_img = keras.Input(shape=(784,))

#encoded  is the encoded representation of the input
encoded = layers.Dense(encoding_dim, activation='relu')(input_img)

#decoded is the lossy reconstruction of the input
decoded = layers.Dense(784, activation='sigmoid')(encoded)




#model maps input to its reconstruction
autoencoder = keras.Model(input_img, decoded)

#seperate encoder model
#maps an input to its encoded representation
encoder = keras.Model(input_img, encoded)
#encoded (32-dimensional) input
encoded_input = keras.Input(shape=(encoding_dim,))

#retrive last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
#create decoder model
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))


#configure model to use a per-pixel binary crossentropy loss, and the Adam optimizer
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

#importing mnist digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(type(x_train))

x_test_temp = x_test 

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

x_test7 = np.array(test_data7)
x_test7 = x_test7.astype('float32')/255
x_test7 = x_test7.reshape((len(x_test7)), np.prod(x_test7.shape[1:]))


x_test_temp = x_test_temp.astype('float32')/255
x_test_temp = x_test_temp.reshape(
    (len(x_test_temp)), np.prod(x_test_temp.shape[1:]))


print(len(x_train))
#normalize all values between 0 and 1 and we will flatten the 28X28 images into vectors of size 784
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
x_train = x_train.reshape((len(x_train)), np.prod(x_train.shape[1:]))
x_test = x_test.reshape((len(x_test)), np.prod(x_test.shape[1:]))
print(x_train.shape)
print(x_test.shape)



#trainning of autoencoder for certain epoch
autoencoder.fit(x_train, x_train,
                epochs=250,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))


#encode and decode some inputs
#note take from test set

encoded_imgs = encoder.predict(x_test7)
decoded_imgs = decoder.predict(encoded_imgs)

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
    plt.imshow(x_test7[i].reshape(28, 28))
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

from skimage.metrics import structural_similarity as ssim


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
error_value = mse(imageA,imageB )
print(error_value)

for i in range(10):
    imageA = x_test7[i].reshape(28, 28)
    imageB = decoded_imgs[i].reshape(28, 28)
    compare_images(imageA, imageB,"Compare")
