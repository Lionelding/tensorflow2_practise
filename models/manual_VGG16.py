import tensorflow as tf
from tensorflow.keras import models, regularizers
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, Dense, MaxPooling2D, Flatten


class VGG16(tf.keras.models.Model):
    def __init__(self, input_shape):

        super(VGG16, self).__init__()

        weight_decay = 0.00
        self.num_classes = 10
        self.dropout_rate = 0.3

        model = models.Sequential()
        model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape, kernel_regularizer=regularizers.l2(weight_decay), activation=tf.nn.relu))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))

        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation=tf.nn.relu))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation=tf.nn.relu))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation=tf.nn.relu))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation=tf.nn.relu))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation=tf.nn.relu))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation=tf.nn.relu))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation=tf.nn.relu))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation=tf.nn.relu))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation=tf.nn.relu))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation=tf.nn.relu))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation=tf.nn.relu))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation=tf.nn.relu))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay), activation=tf.nn.relu))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))

        self.model = model

    def call(self, x):
        y = self.model(x)
        return y
