#Для створення згорткової мережі використовується схема Vanilla Unet.
class CreateVanilla:
    
    def __init__(self, input_layer=Input(shape=(256, 256, 3)), in_m=8, kernel_size=(3, 3), drop_out=0.5):
        self.input_layer_ = input_layer
        self.in_ = in_m
        self.kernel_size_ = kernel_size
        self.drop_out_ = drop_out
    
    @staticmethod
    def compilated(model):
        def dice_coefficient(y_true, y_pred):
            numerator = 2 * tf.reduce_sum(y_true * y_pred)
            denominator = tf.reduce_sum(y_true + y_pred)
        
            return numerator / (denominator + tf.keras.backend.epsilon())

        def loss_dice(y_true, y_pred):
            return binary_crossentropy(y_true, y_pred) - tf.math.log(dice_coefficient(y_true, y_pred) + tf.keras.backend.epsilon())
    
        best_w = keras.callbacks.ModelCheckpoint('vanil_unet_best.hdf5',
                                        monitor='val_loss',
                                        verbose=1,
                                        save_best_only=True,
                                        save_weights_only=True,
                                        mode='auto',
                                        save_freq='epoch')
        stop = EarlyStopping(patience=5, verbose=1)
        callbacks = [best_w, stop]
        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model.compile(adam, loss=loss_dice, metrics=[dice_coefficient])

        finished_model = model.compile(adam, loss=loss_dice, metrics=[dice_coefficient])
        
        return finished_model, callbacks
    
    
    def create_model(self):
        conv_1 = Conv2D(in_ * 1, kernel_size, activation="relu", padding="same")(input_layer)
        conv_1 = Conv2D(in_ * 1, kernel_size, activation="relu", padding="same")(conv_1)
        pool_1 = MaxPooling2D((2, 2))(conv_1)
        pool_1 = Dropout(drop_out)(pool_1)
        
        conv_2 = Conv2D(in_ * 2, kernel_size, activation="relu", padding="same")(pool_1)
        conv_2 = Conv2D(in_ * 2, kernel_size, activation="relu", padding="same")(conv_2)
        pool_2 = MaxPooling2D((2, 2))(conv_2)
        pool_2 = Dropout(drop_out)(pool_2)
        
        conv_3 = Conv2D(in_ * 4, kernel_size, activation="relu", padding="same")(pool_2)
        conv_3 = Conv2D(in_ * 4, kernel_size, activation="relu", padding="same")(conv_3)
        pool_3 = MaxPooling2D((2, 2))(conv_3)
        pool_3 = Dropout(drop_out)(pool_3)
        
        conv_4 = Conv2D(in_ * 8, kernel_size, activation="relu", padding="same")(pool_3)
        conv_4 = Conv2D(in_ * 8, kernel_size, activation="relu", padding="same")(conv_4)
        pool_4 = MaxPooling2D((2, 2))(conv_4)
        pool_4 = Dropout(drop_out)(pool_4)
        
        conv_5 = Conv2D(in_ * 16, kernel_size, activation="relu", padding="same")(pool_4)
        conv_5 = Conv2D(in_ * 16, kernel_size, activation="relu", padding="same")(conv_5)
        pool_5 = MaxPooling2D((2, 2))(conv_5)
        pool_5 = Dropout(drop_out)(pool_5)
        
        conv_6 = Conv2D(in_ * 32, kernel_size, activation="relu", padding="same")(pool_5)
        conv_6 = Conv2D(in_ * 32, kernel_size, activation="relu", padding="same")(conv_6)
        
        trans_5 = Conv2DTranspose(in_ * 16, kernel_size, strides=(2, 2), padding="same")(conv_6)
        conv_conc_5 = concatenate([trans_5, conv_5])
        conv_trans_5 = Dropout(drop_out)(conv_conc_5)
        conv_trans_5 = Conv2D(in_ * 16, kernel_size, activation="relu", padding="same")(conv_trans_5)
        conv_trans_5 = Conv2D(in_ * 16, kernel_size, activation="relu", padding="same")(conv_trans_5)
        
        trans_4 = Conv2DTranspose(in_ * 8, kernel_size, strides=(2, 2), padding="same")(conv_trans_5)
        conv_conc_4 = concatenate([trans_4, conv_4])
        conv_trans_4 = Dropout(drop_out)(conv_conc_4)
        conv_trans_4 = Conv2D(in_ * 8, kernel_size, activation="relu", padding="same")(conv_trans_4)
        conv_trans_4 = Conv2D(in_ * 8, (3, 3), activation="relu", padding="same")(conv_trans_4)
        
        trans_3 = Conv2DTranspose(in_ * 4, kernel_size, strides=(2, 2), padding="same")(conv_trans_4)
        conv_conc_3 = concatenate([trans_3, conv_3])
        conv_trans_3 = Dropout(drop_out)(conv_conc_3)
        conv_trans_3 = Conv2D(in_ * 4, kernel_size, activation="relu", padding="same")(conv_trans_3)
        conv_trans_3 = Conv2D(in_ * 4, kernel_size, activation="relu", padding="same")(conv_trans_3)
        
        trans_2 = Conv2DTranspose(in_ * 2, kernel_size, strides=(2, 2), padding="same")(conv_trans_3)
        conv_conc_2 = concatenate([trans_2, conv_2])
        conv_trans_2 = Dropout(drop_out)(conv_conc_2)
        conv_trans_2 = Conv2D(in_ * 2, kernel_size, activation="relu", padding="same")(conv_trans_2)
        conv_trans_2 = Conv2D(in_ * 2, kernel_size, activation="relu", padding="same")(conv_trans_2)
        
        trans_1 = Conv2DTranspose(in_ * 1, kernel_size, strides=(2, 2), padding="same")(conv_trans_2)
        conv_conc_1 = concatenate([trans_1, conv_1])
        conv_trans_1 = Dropout(drop_out)(conv_conc_1)
        conv_trans_1 = Conv2D(in_ * 1, kernel_size, activation="relu", padding="same")(conv_trans_1)
        conv_trans_1 = Conv2D(in_ * 1, kernel_size, activation="relu", padding="same")(conv_trans_1)
        
        output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(conv_trans_1)
        
        vanil_unet = Model(self.input_layer_, output_layer)
        
        return vanil_unet