import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


def get_dataset(batch_size):
    (x, y), _ = datasets.mnist.load_data()
    print(f'dataset: {x.shape}, {y.shape}, {x.min}, {y.min}')

    x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.batch(batch_size).repeat(10)

    return dataset


def get_model():
    model = Sequential([
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(10)
    ])

    model.build(input_shape=(None, 28*28))
    model.summary()
    return model


def main():

    batch_size =32
    dataset = get_dataset(batch_size)
    model = get_model()

    optimizer = optimizers.SGD(lr=0.01)
    accuracy = metrics.Accuracy()

    for step, (x, y) in enumerate(dataset):
        with tf.GradientTape() as tape:
            x = tf.reshape(x, (-1, 28*28))
            pred = model(x)
            y_hot = tf.one_hot(y, depth=10)
            loss = tf.square(pred-y_hot)

            loss = tf.reduce_sum(loss) / batch_size

        accuracy.update_state(tf.argmax(pred, axis=1), y)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 200 ==0:
            print(f'step: {step}, loss: {float(loss)}, acc: {accuracy.result().numpy()}')
            accuracy.reset_states()


if __name__ == '__main__':
    main()