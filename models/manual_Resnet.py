import tensorflow as tf


def conv3x3(channels, strides=1, kernel=(3, 3)):
    return tf.keras.layers.Conv2D(channels, kernel, strides=strides, padding='same', use_bias=False, kernel_initializer=tf.random_normal_initializer())


class ResnetBlock(tf.keras.Model):
    def __init__(self, channels, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()

        self.channels = channels
        self.strides = strides
        self.residual_path = residual_path

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = conv3x3(self.channels, self.strides)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = conv3x3(self.channels)

        if residual_path:
            self.down_bn = tf.keras.layers.BatchNormalization()
            self.down_conv = conv3x3(self.channels, self.strides, kernel=(1, 1))

    def call(self, inputs, training=None):
        residual = inputs
        x = self.bn1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        if self.residual_path:
            residual = self.down_bn(inputs, training=training)
            residual = tf.nn.relu(residual)
            residual = self.down_conv(residual)

        x = x + residual
        return x


class Resnet(tf.keras.Model):
    def __init__(self, block_list, num_classes, initial_filters=16, **kwargs):
        super(Resnet, self).__init__()

        self.block_list = block_list
        self.num_classes = num_classes
        self.initial_filters = initial_filters

        self.in_channels = initial_filters
        self.out_channels = initial_filters
        self.conv_initial = conv3x3(self.out_channels)
        self.blocks = tf.keras.models.Sequential(name='dynamic-blocks')

        for block_id, _ in enumerate(block_list):
            for layer_id in range(block_list[block_id]):
                if block_id != 0 and layer_id == 0:
                    block = ResnetBlock(self.out_channels, strides=2, residual_path=True)

                else:
                    if self.in_channels != self.out_channels:
                        residual_path = True
                    else:
                        residual_path = False
                    block = ResnetBlock(self.out_channels, residual_path=residual_path)

                self.in_channels = self.out_channels
                self.blocks.add(block)

            self.out_channels = self.out_channels * 2

        self.final_bn = tf.keras.layers.BatchNormalization()
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=None):
        outputs = self.conv_initial(inputs)
        outputs = self.blocks(outputs, training=training)
        outputs = self.final_bn(outputs, training=training)
        outputs = tf.nn.relu(outputs)

        outputs = self.avg_pool(outputs)
        outputs = self.fc(outputs)

        return outputs