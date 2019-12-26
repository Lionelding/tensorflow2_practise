import os
import numpy as np
import tensorflow as tf

EPOCHS = 100


class Regressor(tf.keras.layers.Layer):
    def __init__(self):
        super(Regressor, self).__init__()

        # Shape must be specified!

        self.w = self.add_variable('w', [13, 1])
        self.b = self.add_variable('b', [1])

        print(self.w.shape, self.b.shape)
        print(type(self.w), self.w.name, tf.is_tensor(self.w))
        print(type(self.b), self.b.name, tf.is_tensor(self.b))

    def call(self, x):
        y = tf.matmul(x, self.w) + self.b

        return y


def train_and_evaluate(model, optimizer, criteon, train_db, valid_db):
    for epoch in range(EPOCHS):
        for batch_id, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                logits = model(x)
                logits = tf.squeeze(logits, axis=1)
                loss = criteon(logits, y)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        print(f'epoch: {epoch}, batch: {batch_id}, loss: {loss.numpy()}')

        if epoch % 10 == 0:
            for x, y in valid_db:

                logits = model(x)
                logits = tf.squeeze(logits, axis=1)
                loss = criteon(logits, y)

                print(f'epoch: {epoch}, loss: {loss.numpy()}')


def main():
    tf.random.set_seed(1234)
    np.random.seed(1234)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print(tf.__version__)

    (x_train, y_train), (x_valid, y_valid) = tf.keras.datasets.boston_housing.load_data()

    x_train, x_valid = x_train.astype(np.float32), x_valid.astype(np.float32)
    print(x_train.shape, y_train.shape)
    print(x_valid.shape, y_valid.shape)

    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(64)
    valid_db = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(64)

    model = Regressor()
    # import pdb;pdb.set_trace()

    criteon = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

    train_and_evaluate(model, optimizer, criteon, train_db, valid_db)
    print("lol")


if __name__ == '__main__':
    main()
