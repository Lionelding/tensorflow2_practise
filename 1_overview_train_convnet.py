import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets, metrics


MODEL_DIR = '/tmp/tensorflow/mnist'
NUM_TRAIN_EPOCHS = 5


def mnist_dataset():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train, x_test = x_train / np.float32(255), x_test / np.float32(255)
    y_train, y_test = y_train.astype(np.int64), y_test.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    return train_dataset, test_dataset


def train_step(model, optimizer, images, labels):
    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss = compute_loss(labels, logits)
        compute_accuracy(labels, logits)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


def train(model, optimizer, dataset, log_freq=50):
    avg_loss = metrics.Mean('loss', dtype=tf.float32)

    for images, labels in dataset:
        loss = train_step(model, optimizer, images, labels)
        avg_loss(loss)

    if tf.equal(optimizer.iterations % log_freq, 0):
        print('step: ', int(optimizer.iterations),
              'loss: ', avg_loss.result().numpy(),
              'acc: ', compute_accuracy.result().numpy())
        avg_loss.reset_states()
        compute_accuracy.reset_states()


def test(model, dataset, step_num):
    avg_loss = metrics.Mean('loss', dtype=tf.float32)
    for (images, labels) in datasets:
        logits = model(images, training=False)
        avg_loss(compute_loss(labels, logits))
        compute_accuracy(labels, logits)

    print('Model test set loss: {:0.4f} accuracy: {:0.2f}%'.format(
        avg_loss.result(), compute_accuracy.result() * 100))

    print('loss:', avg_loss.result(), 'acc:', compute_accuracy.result())


def get_model():

    model = keras.Sequential([
        layers.Reshape(target_shape=[28, 28, 1], input_shape=(28, 28,)),
        layers.Conv2D(2, 5, padding='same', activation=tf.nn.relu),
        layers.MaxPooling2D((2, 2), (2, 2), padding='same'),
        layers.Conv2D(4, 5, padding='same', activation=tf.nn.relu),
        layers.MaxPooling2D((2, 2), (2, 2), padding='same'),
        layers.Flatten(),
        layers.Dense(32, activation=tf.nn.relu),
        layers.Dropout(rate=0.4),
        layers.Dense(10)
    ])

    return model


def apply_clean():
    if tf.io.gfile.exists(MODEL_DIR):
        print(f'Removing: {MODEL_DIR}')
        tf.io.gfile.rmtree(MODEL_DIR)

compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
compute_accuracy = tf.keras.metrics.SparseCategoricalCrossentropy()


def main():

    # Dataset
    train_ds, test_ds = mnist_dataset()
    train_ds = train_ds.shuffle(60000).batch(64)
    test_ds = test_ds.batch(32)

    # Create a model and an optimizer
    model = get_model()
    optimizer = optimizers.SGD(learning_rate=0.01, momentum=0.5)

    apply_clean()
    checkpoint_dir = os.path.join(MODEL_DIR, 'checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    for i in range(NUM_TRAIN_EPOCHS):
        start = time.time()

        train(model, optimizer, train_ds, log_freq=500)
        end = time.time()

        print(f'Epoch: {(i + 1)}, Steps: {int(optimizer.iterations)} Time: {end - start}')
        checkpoint.save(checkpoint_prefix)
        print('saved checkpoint.')

    export_path = os.path.join(MODEL_DIR, 'export')
    tf.saved_model.save(model, export_path)
    print('DONE')


if __name__ == '__main__':
    main()