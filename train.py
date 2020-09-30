import tensorflow.keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf

from tensorflow.keras.utils import to_categorical

# config = tf.ConfigProto(device_count={'GPU': 0, 'CPU': 56})
# sess = tf.Session(config=config)
# tensorflow.keras.backend.set_session(sess)


def reshape_df(df):
    for i in range(len(df)):
        df[i] = df[i] / 255
        df[i] = df[i].reshape(48, 48, 1)
    return np.array(df)


def reshape_target(tg):
    result = np.array(tg)
    result = result.reshape(result.shape[0], 1)
    return to_categorical(result)


class facialEmotionRecognition:

    def __init__(self):
        self.model = None

    def buildModel(self, num_classes):
        self.model = Sequential()

        # 1st convolution layer
        self.model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
        self.model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

        # 2nd convolution layer
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

        # 3rd convolution layer
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

        self.model.add(Flatten())

        # fully connected neural networks
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(num_classes, activation='softmax'))

        # loss = SparseCategoricalCrossentropy()
        # list_metrics = [SparseCategoricalAccuracy()]
        self.model.compile(loss='categorical_crossentropy'
                      , optimizer=tensorflow.keras.optimizers.Adam()
                      , metrics=['accuracy'])
        print(self.model.summary())

    def train_fit(self, X_train, y_train, batch_size, epochs):
        gen = ImageDataGenerator()
        es = EarlyStopping()
        train_generator = gen.flow(X_train, y_train, batch_size=batch_size)
        fit = True
        if fit == True:
            self.model.fit_generator(train_generator,
                                     steps_per_epoch=batch_size,
                                     epochs=epochs)
        else:
            self.model.load_weights('/data/facial_expression_model_weights.h5')

    def save_model(self):
        self.model.save('model/model_1.hdf5')

    def load_model(self):
        self.model = load_model('model/model_1.hdf5')