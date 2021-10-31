import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, GlobalAveragePooling2D, Dropout, UpSampling2D, Conv2D, MaxPooling2D, Activation

from skimage.transform import resize
from keras import backend as K

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


path = 'C:/Users/Vladislav/Desktop/Airbus'
path_to_train = 'C:/Users/Vladislav/Desktop/Airbus/train_v2'
os.chdir(path)

def keras_generator(df, batch_size):
    while True:
        x_batch = []
        y_batch = []

        for i in range(batch_size):
            img, mask = df.sample(1).values[0]

            img = cv2.resize(img, (256, 256))
            mask = cv2.resize(mask, (256, 256))

            x_batch += [img]
            y_batch += [mask]

        x_batch = np.array(x_batch) / 255
        y_batch = np.array(y_batch)

        yield x_batch, np.expand_dims(y_batch, -1)


inp = Input(shape=(256, 256, 3))

conv_1_1 = Conv2D(32, (3, 3), padding='same')(inp)
conv_1_1 = Activation('relu')(conv_1_1)

conv_1_2 = Conv2D(32, (3, 3), padding='same')(conv_1_1)
conv_1_2 = Activation('relu')(conv_1_2)

pool_1 = MaxPooling2D(2)(conv_1_2)

conv_2_1 = Conv2D(64, (3, 3), padding='same')(pool_1)
conv_2_1 = Activation('relu')(conv_2_1)

conv_2_2 = Conv2D(64, (3, 3), padding='same')(conv_2_1)
conv_2_2 = Activation('relu')(conv_2_2)

pool_2 = MaxPooling2D(2)(conv_2_2)

conv_3_1 = Conv2D(128, (3, 3), padding='same')(pool_2)
conv_3_1 = Activation('relu')(conv_3_1)

conv_3_2 = Conv2D(128, (3, 3), padding='same')(conv_3_1)
conv_3_2 = Activation('relu')(conv_3_2)

conv_3_3 = Conv2D(128, (3, 3), padding='same')(conv_3_2)
conv_3_3 = Activation('relu')(conv_3_3)

pool_3 = MaxPooling2D(2)(conv_3_3)

conv_4_1 = Conv2D(256, (3, 3), padding='same')(pool_3)
conv_4_1 = Activation('relu')(conv_4_1)

conv_4_2 = Conv2D(256, (3, 3), padding='same')(conv_4_1)
conv_4_2 = Activation('relu')(conv_4_2)

conv_4_3 = Conv2D(256, (3, 3), padding='same')(conv_4_2)
conv_4_3 = Activation('relu')(conv_4_3)

pool_4 = MaxPooling2D(2)(conv_4_3)

up_1 = UpSampling2D(2, interpolation='bilinear')(pool_4)
conc_1 = Concatenate()([conv_4_3, up_1])

conv_up_1_1 = Conv2D(256, (3, 3), padding='same')(conc_1)
conv_up_1_1 = Activation('relu')(conv_up_1_1)

conv_up_1_2 = Conv2D(256, (3, 3), padding='same')(conv_up_1_1)
conv_up_1_2 = Activation('relu')(conv_up_1_2)

conv_up_1_3 = Conv2D(256, (3, 3), padding='same')(conv_up_1_2)
conv_up_1_3 = Activation('relu')(conv_up_1_3)

up_2 = UpSampling2D(2, interpolation='bilinear')(conv_up_1_3)
conc_2 = Concatenate()([conv_3_3, up_2])

conv_up_2_1 = Conv2D(128, (3, 3), padding='same')(conc_2)
conv_up_2_1 = Activation('relu')(conv_up_2_1)

conv_up_2_2 = Conv2D(128, (3, 3), padding='same')(conv_up_2_1)
conv_up_2_2 = Activation('relu')(conv_up_2_2)

conv_up_2_3 = Conv2D(128, (3, 3), padding='same')(conv_up_2_2)
conv_up_2_3 = Activation('relu')(conv_up_2_3)

up_3 = UpSampling2D(2, interpolation='bilinear')(conv_up_2_3)
conc_3 = Concatenate()([conv_2_2, up_3])

conv_up_3_1 = Conv2D(64, (3, 3), padding='same')(conc_3)
conv_up_3_1 = Activation('relu')(conv_up_3_1)

conv_up_3_2 = Conv2D(64, (3, 3), padding='same')(conv_up_3_1)
conv_up_3_2 = Activation('relu')(conv_up_3_2)

conv_up_3_3 = Conv2D(64, (3, 3), padding='same')(conv_up_3_2)
conv_up_3_3 = Activation('relu')(conv_up_3_3)

up_4 = UpSampling2D(2, interpolation='bilinear')(conv_up_3_3)
conc_4 = Concatenate()([conv_1_2, up_4])

conv_up_4_1 = Conv2D(32, (3, 3), padding='same')(conc_4)
conv_up_4_1 = Activation('relu')(conv_up_4_1)

conv_up_4_2 = Conv2D(32, (3, 3), padding='same')(conv_up_4_1)
conv_up_4_2 = Activation('relu')(conv_up_4_2)

conv_up_4_3 = Conv2D(1, (3, 3), padding='same')(conv_up_4_2)
result = Activation('sigmoid')(conv_up_4_3)


model = Model(inputs=inp, outputs=result)

vanil_unet = Model(inputs=inp, outputs=result)


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    return K.mean((2. * intersection + smooth) / (union + smooth))


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


best_w = keras.callbacks.ModelCheckpoint('vanil_unet_best.hdf5',
                                monitor='val_loss',
                                verbose=0,
                                save_best_only=True,
                                save_weights_only=True,
                                mode='auto',
                                period=1)

last_w = keras.callbacks.ModelCheckpoint('vanil_unet_last.hdf5',
                                monitor='val_loss',
                                verbose=0,
                                save_best_only=False,
                                save_weights_only=True,
                                mode='auto',
                                period=1)

callbacks = [best_w, last_w]

adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

vanil_unet.compile(adam, 'binary_crossentropy', metrics=['acc', dice_coef_loss])


vanil_unet.summary()


tf.random.set_seed(5)

batch_size = 20

vanil_unet.fit_generator(keras_generator(df_train, batch_size),
              steps_per_epoch=50,
              epochs=10,
              callbacks=callbacks,
              verbose=1,
              validation_data=keras_generator(df_valid, batch_size),
              validation_steps=30,
              class_weight=None,
              max_queue_size=10,
              workers=1,
              use_multiprocessing=False,
              shuffle=True,
              initial_epoch=0)


vanil_unet.save(path)