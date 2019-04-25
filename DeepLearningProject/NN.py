import tensorflow as tf
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array
from keras import backend as K
import os
import h5py
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

K.set_image_dim_ordering('th')

import numpy as np

import os
from PIL import Image
import theano
from pandas import HDFStore


theano.config.optimizer = "None"
from sklearn.model_selection import train_test_split

# input image dimensions
m, n = 48, 48

path1 = r'C:\Users\Wahba\Desktop\test'
path2 = r'C:\Users\Wahba\Desktop\train'

classes = os.listdir(path2)
print(classes)
x = []
y = []
for fol in classes:
    print(fol)
    imgfiles = os.listdir(path2+'\\' +fol)
    for img in imgfiles:
        im = Image.open(path2 + '\\' + fol + '\\' + img);
        im = im.convert(mode='RGB')
        imrs = im.resize((m, n))
        imrs = img_to_array(imrs) / 255;
        imrs = imrs.transpose(2, 0, 1);
        imrs = imrs.reshape(3, m, n);
        x.append(imrs)
        y.append(fol)

x = np.array(x);
y = np.array(y);

print(x)
print(y)

nb_classes = len(classes)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

uniques, id_train = np.unique(y_train, return_inverse=True)
Y_train = np_utils.to_categorical(id_train, nb_classes)
uniques, id_test = np.unique(y_test, return_inverse=True)
Y_test = np_utils.to_categorical(id_test, nb_classes)



model = tf.keras.models.Sequential([
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(128, activation=tf.nn.sigmoid),
tf.keras.layers.Dense(128, activation=tf.nn.sigmoid),

tf.keras.layers.Dropout(0.5),
tf.keras.layers.Dense((nb_classes), activation=tf.nn.softmax)
])

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

model.fit(x_train, Y_train, batch_size=5, epochs=15, verbose=1, validation_data=(x_test, Y_test))
model.evaluate(x_test, Y_test)
model.save_weights('NN.h5')


files = os.listdir(path1);
for i in range(len(files)):
    img = files[i]
    # img = input()
    print(img)
    im = Image.open(path1 + '\\' + img);
    imrs = im.resize((m, n))
    imrs = img_to_array(imrs) / 255;
    imrs = imrs.transpose(2, 0, 1);
    imrs = imrs.reshape(3, m, n);

    x = []
    x.append(imrs)
    x = np.array(x);
    predictions = model.predict(x)
    #print(predictions)
    print(classes[np.argmax(predictions)])
