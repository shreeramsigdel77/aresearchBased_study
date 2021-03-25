from keras.datasets import mnist
import matplotlib.pyplot as plt



(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(y_test)

train_data = []
for count,value in enumerate(y_train):
    if value == 1:
        train_data.append(x_train[count])


n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    #Display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(train_data[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #Display reconstruction
    # ax = plt.subplot(2, n, i + 1 + n)
    # plt.imshow(decoded_imgs[i].reshape(28, 28))
    # plt.gray()
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
plt.show()
