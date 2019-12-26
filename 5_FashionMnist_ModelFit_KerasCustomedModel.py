import os
import tensorflow as tf
from tensorflow.keras import layers, optimizers


def prepare_mnist_features_and_labels(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y


def get_datasets():
    (x_train, y_train), (x_valid, y_valid) = tf.keras.datasets.mnist.load_data()
    print(x_train.shape, y_train.shape)
    print(x_valid.shape, y_valid.shape)

    y_train, y_valid = tf.one_hot(y_train, depth=10), tf.one_hot(y_valid, depth=10)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.map(prepare_mnist_features_and_labels)
    train_ds = train_ds.shuffle(1000).batch(64)

    valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    valid_ds = valid_ds.map(prepare_mnist_features_and_labels)
    valid_ds = valid_ds.batch(64)

    return train_ds, valid_ds


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

        self.layer1 = layers.Dense(200, activation=tf.nn.relu)
        self.layer2 = layers.Dense(200, activation=tf.nn.relu)
        self.layer3 = layers.Dense(128, activation=tf.nn.relu)
        self.layer4 = layers.Dense(10, activation=tf.nn.relu)

    def call(self, x, training=False):
        x = tf.reshape(x, [-1, 28*28])
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def main():
    tf.random.set_seed(1234)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    train_ds, valid_ds = get_datasets()

    model = MyModel()

    optimizer = optimizers.Adam(learning_rate=1e-2)
    loss_func = tf.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer,
                  loss=loss_func,
                  metrics=['accuracy'])

    model.fit(train_ds.repeat(),
              epochs=300,
              steps_per_epoch=500,
              validation_data=valid_ds.repeat(),
              validation_steps=2)


if __name__ == '__main__':
    main()
