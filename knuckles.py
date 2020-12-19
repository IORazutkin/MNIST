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

def get_data(model, image_dir):
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
            true.append(int(list[i]) // 10)

    test = np.asarray(test)
    test = 1 - test / 255.0
    test = test.reshape((len(test), 28, 28, 1))
    true = np.asarray(true)
    return test, true

# model = Sequential([
#     Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1)),
#     MaxPooling2D(pool_size=(2, 2)),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(4, activation='softmax')
# ])
#
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

model = tf.keras.models.load_model('models/knuckles')
x_train, y_train = get_data(model, 'C:/Users/igorr/PycharmProjects/AI/train_knuckles_dataset')
# model.fit(x=x_train, y=y_train, epochs=20)
# model.save('models/knuckles')

x_test, y_test = get_data(model, 'C:/Users/igorr/PycharmProjects/AI/my_knuckles_dataset')

model.evaluate(x=x_test, y=y_test)

predict = model.predict_classes(x_test)
# print(predict)

# image_index = 800
# plt.imshow(x_train[image_index].reshape(160, 160), cmap='Greys')
# pred = model.predict(x_train[image_index].reshape(1, 160, 160, 1))
# print(pred.argmax(), y_train[image_index])
# plt.show()

con_mat = tf.math.confusion_matrix(labels=y_test, predictions=predict).numpy()
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

classes=[0, 1, 2, 3]

figure = plt.figure(figsize=(8, 8))
seaborn.heatmap(con_mat_norm, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
