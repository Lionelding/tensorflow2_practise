import timeit
import tensorflow as tf

cell = tf.keras.layers.LSTMCell(10)

@tf.function
def func(input, state):
    return cell(input, state)

input = tf.zeros([10, 10])
state = [tf.zeros([10, 10])] * 2

cell(input, state)
func(input, state)

dynamic_graph_time = timeit.timeit(lambda: cell(input, state), number=100)
static_graph_time = timeit.timeit(lambda: func(input, state), number=100)
print(f'dynamic_graph_time: {dynamic_graph_time}')
print(f'static_graph_time: {static_graph_time}')