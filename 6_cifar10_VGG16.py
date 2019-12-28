import os
import numpy as np
import tensorflow as tf
from models.manual_VGG16 import VGG16


def prepare_cifar10(data, labels):

    data = tf.cast(data, tf.float32) / 255.
    labels = tf.cast(labels, tf.int32)

    # import pdb;pdb.set_trace()

    # mean = np.mean(data, axis=(0, 1, 2, 3))
    # std = np.std(data, axis=(0, 1, 2, 3))

    return data, labels


def train_and_evaluate(model, criteon, optimizer, metric, train_ds, valid_ds, epochs):

    for epoch in range(epochs):
        for batch_id, (x, y) in enumerate(train_ds):
            y = tf.squeeze(y, axis=1)
            y = tf.one_hot(y, depth=10)

            with tf.GradientTape() as tape:
                logits = model(x)
                train_loss = criteon(y, logits)
                metric.update_state(y, logits)

            grads = tape.gradient(train_loss, model.trainable_variables)
            grads = [tf.clip_by_norm(g, 15) for g in grads]
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            print(f'epoch: {epoch}, batch_id: {batch_id}, train_loss: {train_loss}, train_acc: {metric.result().numpy()}')

        metric.reset_states()

        if epoch % 10 ==0:
            for x, y in valid_ds:
                y = tf.squeeze(y, axis=1)
                y = tf.one_hot(y, depth=10)

                logits = model.predict(x)
                valid_loss = criteon(y, logits)
                metric.update_state(y, logits)

            print(f'epoch: {epoch}, batch_id: {batch_id}, valid_loss: {valid_loss}, valid_acc: {metric.result().numpy()}')
            metric.reset_states()


def main():
    tf.random.set_seed(1234)

    (x_train, y_train), (x_valid, y_valid) = tf.keras.datasets.cifar10.load_data()
    print(x_train.shape, y_train.shape)
    print(x_valid.shape, y_valid.shape)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.map(prepare_cifar10)
    train_ds = train_ds.shuffle(1000).batch(64)

    valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    valid_ds = valid_ds.map(prepare_cifar10)
    valid_ds = valid_ds.batch(64)

    model = VGG16([32, 32, 3])
    criteon = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.CategoricalAccuracy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    train_and_evaluate(model, criteon, optimizer, metric, train_ds, valid_ds, 100)

    return


if __name__ == '__main__':
    main()