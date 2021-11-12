#Імпортуємо усі необхіжні бібліотеки

import os

from tqdm import tqdm

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import _pickle as cPickle

import cv2

from scipy.ndimage import rotate

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from tensorflow.keras.layers import Conv2DTranspose
from keras.layers.merge import concatenate
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Activation
from keras.callbacks import EarlyStopping

from keras import backend as K
from keras.losses import binary_crossentropy

from create_vanilla import CreateVanilla

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#Встановлюємо директорію (наприклад, моя директорія, де все розміщене).
path = 'C:/Users/Vladislav/Desktop/Airbus'
path_to_train = 'C:/Users/Vladislav/Desktop/Airbus/train_v2'
os.chdir(path)

#Створюємо додаткову функцію, яка перетворить файли у необхідний формат 
#для нейромережі.
#--->

#Встановлюємо розмір зображення для навчання мережі.
size = (256, 256)

#Функція, яка трансформуватиме зображення та маски у масиви заданого розміру.
def keras_transform(df):
    lenght = df.shape[0]
    x = []
    y = []
      
    for i in tqdm(range(lenght)):
        img = df.iloc[i, 0]
        img = cv2.resize(img, size)
        
        mask = df.iloc[i, 1]
        mask = cv2.resize(mask, size)
        mask = rotate(mask, angle=90)
        mask = np.flip(mask, 0)
        
        x.append(img)
        y.append(mask)

    x = np.array(x) / 255
    y = np.array(y)
    
    return x, np.expand_dims(y, -1)

#<---

#Завантажуємо готові файли для перетворення та використання в навчанні.
df_train = pd.read_pickle('train_set.pkl')
df_valid = pd.read_pickle('valid_set.pkl')
df_test = pd.read_pickle('test_set.pkl')

#Перетоврюємо у потрібний нам формат
X_train, y_train = keras_transform(df_train)
y_train = y_train.astype(np.float32) 

X_valid, y_valid = keras_transform(df_valid)
y_valid = y_valid.astype(np.float32)

X_test, y_test = keras_transform(df_test)
y_test = y_test.astype(np.float32) 

#Візуалізуємо для перевірки вмісту
img_id = 7

fig, axs = plt.subplots(1, 2, figsize=(20, 20))

axs[0].imshow(X_train[img_id])
axs[1].imshow(y_train[img_id])

plt.show()

#Завантажуємо нашу підготовлену модель
vanil_unet = CreateVanilla()

#Створюємо архітектуру мережі
vanil_unet = CreateVanilla.create_vanilla(vanil_unet)

#Дивимося на архітектуру
vanil_unet.summary()

#Компілюємо модель
vanil_unet, callbacks = CreateVanilla.compilated(vanil_unet)

#Далі ми можемо навчати нашу створену моддель на наявних даних
# tf.config.experimental_run_functions_eagerly(True)
# tf.random.set_seed(5)

# results = vanil_unet.fit(X_train, 
#               y_train,
#               batch_size=20,
#               steps_per_epoch=40,
#               epochs=5,
#               callbacks=callbacks,
#               verbose=1,
#               validation_data=(X_valid, y_valid),
#               validation_steps=20,
#               shuffle=True) 

#Я пропоную одразу завантажити найкращі параметри перед тим отриманні під час навчання,
#а також результати даного навчання
vanil_unet.load_weights('vanil_unet_best.hdf5')

#Результати прогресу були втрачені після невдалого повторого запуску моделі. Прошу вибачення.
#Зрозумів свою помилку.
#Необхідно було робити навчання повторно через цикл for із вимірюванням валідації 
#та збереженням у окремий словник. Параметром циклу слугувало би кількість епох.

#Оцінимо роботу мережі на тестовому наборі даних
vanil_unet.evaluate(X_test, y_test, verbose=1, batch_size=100)

#Візуалізуємо результат передбачення та порівняємо з оригіналом
amount_in_test = 20
preds_test = vanil_unet.predict(X_test[:amount_in_test], verbose=1)

#Параметр для вибору зображення з тестового набору
img_id = 11

fig, axs = plt.subplots(1, 3, figsize=(20, 20))

axs[0].imshow(X_test[img_id])
axs[0].set_title('Image', fontsize=20)

axs[1].imshow(y_test[img_id])
axs[1].set_title('Mask', fontsize=20)

axs[2].imshow(preds_test[img_id, ..., 0] > 0.1)
axs[2].set_title('Predict', fontsize=20)

plt.show()


