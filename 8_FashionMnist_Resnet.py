import os
import numpy as np
import tensorflow as tf
from models.manual_Resnet import Resnet


def main():
    tf.random.set_seed(1234)
    np.random.seed(1234)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    batch_size = 256
    epochs = 100

    (x_train, y_train), (x_valid, y_valid) = tf.keras.datasets.fashion_mnist.load_data()
    x_train, x_valid = tf.cast(x_train, tf.float32) / 255., tf.cast(x_valid, tf.float32) / 255.
    x_train, x_valid = tf.expand_dims(x_train, axis=3), tf.expand_dims(x_valid, axis=3)
    y_train, y_valid = tf.one_hot(y_train, depth=10), tf.one_hot(y_valid, depth=10)

    print(x_train.shape, y_train.shape)
    print(x_valid.shape, y_valid.shape)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(batch_size)
    valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(batch_size)

    model = Resnet([2, 2, 2], 10)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    criteon = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metrics = tf.keras.metrics.Accuracy()

    model.compile(optimizer=optimizer,
                  loss=criteon,
                  metrics=[metrics])

    model.build(input_shape=(None, 28, 28, 1))
    model.summary()

    model.fit(train_ds.repeat(),
              epochs=epochs,
              steps_per_epoch=500,
              validation_data=valid_ds.repeat(),
              validation_steps=2,
              verbose=1)


if __name__ == '__main__':
    main()
