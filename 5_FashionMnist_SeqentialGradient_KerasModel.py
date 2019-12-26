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

    # Don't cast to categorical since the loss function
    # y_train, y_valid = tf.one_hot(y_train, depth=10), tf.one_hot(y_valid, depth=10)

    # Convert numpy to tensor first! The following does not work!
    # y_train, y_valid = y_train.astype(np.float32), y_valid.astype(np.float32)
    # y_train, y_valid = tf.one_hot(y_train, depth=10), tf.one_hot(y_valid, depth=10)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.map(prepare_mnist_features_and_labels)
    train_ds = train_ds.shuffle(1000).batch(64)

    valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    valid_ds = valid_ds.map(prepare_mnist_features_and_labels)
    valid_ds = valid_ds.batch(64)

    return train_ds, valid_ds




def get_model():
    model = tf.keras.Sequential([
        layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
        layers.Dense(200, activation=tf.nn.relu),
        layers.Dense(200, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(10)
    ])

    return model


@tf.function
def compute_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))


@tf.function
def compute_accuracy(logits, labels):
    pred = tf.argmax(logits, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(pred, labels), tf.float32))


def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        # import pdb;
        # pdb.set_trace()

        loss = compute_loss(logits, y)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    accuracy = compute_accuracy(logits, y)

    return loss, accuracy


def evaluate(model, x, y):
    logits = model(x)
    loss = compute_loss(logits, y)
    accuracy = compute_accuracy(logits, y)
    return loss, accuracy


def main():
    epochs = 100
    train_ds, valid_ds = get_datasets()

    model = get_model()
    optimizer = optimizers.Adam(learning_rate=1e-2)

    for epoch in range(epochs):
        for batch_id, (x, y) in enumerate(train_ds):
            train_loss, train_accuracy = train_one_step(model, optimizer, x, y)

            if batch_id % 20 == 0:
                for x, y in valid_ds:
                    valid_loss, valid_accuracy = evaluate(model, x, y)

                print(
                    f'epoch: {epoch}, batch_id: {batch_id}, train loss: {train_loss}, train_accuracy: {train_accuracy}, valid_loss: {valid_loss}, valid_accuracy: {valid_accuracy}')


if __name__ == '__main__':
    main()
