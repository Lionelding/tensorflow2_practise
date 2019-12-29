import os
import ipdb
import numpy as np
import tensorflow as tf
from models.manual_Inception import Inception


def train_and_evaluate(model, optimizer, criteon, metrics, train_ds, valid_ds, batch_size, epochs):
    for epoch in range(epochs):
        for batch_id, (x, y) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                logits = model(x)
                train_loss = criteon(tf.one_hot(y, depth=10), logits)

            grads = tape.gradient(train_loss, model.trainable_variables)
            # ipdb.set_trace()
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            print(f'epoch: {epoch}, batch: {batch_id}, train_loss: {train_loss.numpy()}')

        if epoch % 10 == 0:
            for x, y in valid_ds:
                # ipdb.set_trace()
                logits = model(x)
                predictinos = tf.argmax(logits, axis=1)
                metrics.update_state(y, predictinos)
                valid_loss = criteon(tf.one_hot(y, depth=10), logits)

        print(
            f'epoch: {epoch}, valid_loss: {valid_loss.numpy()}, valid_acc: {metrics.result().numpy()}')
        metrics.reset_states()

    return


def main():
    tf.random.set_seed(1234)
    np.random.seed(1234)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    batch_size = 256
    epochs = 100

    (x_train, y_train), (x_valid, y_valid) = tf.keras.datasets.mnist.load_data()
    x_train, x_valid = tf.cast(x_train, tf.float32) / 255., tf.cast(x_valid, tf.float32) / 255.
    x_train, x_valid = tf.expand_dims(x_train, axis=3), tf.expand_dims(x_valid, axis=3)

    print(x_train.shape, y_train.shape)
    print(x_valid.shape, y_valid.shape)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(batch_size)
    valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(batch_size)

    model = Inception(2, 10)

    model.build(input_shape=(None, 28, 28, 1))
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    criteon = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    metrics = tf.keras.metrics.Accuracy()

    train_and_evaluate(model, optimizer, criteon, metrics, train_ds, valid_ds, batch_size, epochs)


if __name__ == '__main__':
    main()
