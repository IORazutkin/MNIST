import os

import seaborn as seaborn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray, random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.metrics import confusion_matrix

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

# model = Sequential([
#     Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1)),
#     MaxPooling2D(pool_size=(2, 2)),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(10, activation='softmax')
# ])
#
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(x=x_train, y=y_train, epochs=10)

model = tf.keras.models.load_model('models/simple')

print(x_train)
model.evaluate(x_test, y_test)

# image_index = 4444
# plt.imshow(x_test[image_index].reshape(28, 28), cmap='Greys')
# pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
# print(pred.argmax())
# plt.show()

# def mlp_digits_predict(model, image_file):
#     image_size = 28
#     img = tf.keras.preprocessing.image.load_img(image_file, target_size=(image_size, image_size),
#                                                 color_mode='grayscale')
#     img_arr = np.expand_dims(img, axis=0)
#     img_arr = 1 - img_arr / 255.0
#     img_arr = img_arr.reshape((1, 28, 28, 1))
#     result = model.predict_classes(img_arr)
#     return result[0]


def predict(model, image_dir):
    image_size = 28
    list = os.listdir(image_dir)
    test = []
    true = []

    for i in range(0, len(list)):
        dir_name = image_dir + '/' + list[i]
        files = os.listdir(dir_name)
        for j in range(0, len(files)):
            img = tf.keras.preprocessing.image.load_img(dir_name + '/' + files[j],
                                                        target_size=(image_size, image_size),
                                                        color_mode='grayscale')

            test.append(np.expand_dims(img, axis=0)[0])
            true.append(int(list[i]))

    test = np.array(test)
    test = 1 - test / 255.0
    test = test.reshape((len(test), 28, 28, 1))
    result = model.predict_classes(test)
    return result, true


pred, true = predict(model, 'C:/Users/igorr/PycharmProjects/AI/my_dataset')
print(pred)
con_mat = tf.math.confusion_matrix(labels=true, predictions=pred).numpy()

con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

print(con_mat_norm)

classes=[0,1,2,3,4,5,6,7,8,9]

figure = plt.figure(figsize=(8, 8))
seaborn.heatmap(con_mat_norm, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# print("2 - " + str(mlp_digits_predict(model, os.path.abspath("C:/Users/igorr/PycharmProjects/AI/my_dataset/2.jpg"))))
# print("5 - " + str(mlp_digits_predict(model, os.path.abspath("C:/Users/igorr/PycharmProjects/AI/my_dataset/5.jpg"))))
# print("7 - " + str(mlp_digits_predict(model, os.path.abspath("C:/Users/igorr/PycharmProjects/AI/my_dataset/7.jpg"))))
# print("7 - " + str(mlp_digits_predict(model, os.path.abspath("C:/Users/igorr/PycharmProjects/AI/my_dataset/7.1.jpg"))))
# print("7 - " + str(mlp_digits_predict(model, os.path.abspath("C:/Users/igorr/PycharmProjects/AI/my_dataset/7.2.jpg"))))
# print("7 - " + str(mlp_digits_predict(model, os.path.abspath("C:/Users/igorr/PycharmProjects/AI/my_dataset/7.3.jpg"))))
# print("7 - " + str(mlp_digits_predict(model, os.path.abspath("C:/Users/igorr/PycharmProjects/AI/my_dataset/7.4.jpg"))))
# print("9 - " + str(mlp_digits_predict(model, os.path.abspath("C:/Users/igorr/PycharmProjects/AI/my_dataset/9.jpg"))))
