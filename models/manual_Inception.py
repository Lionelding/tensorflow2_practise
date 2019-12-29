import tensorflow as tf


class ConvBnRelu(tf.keras.Model):
    def __init__(self, channel, kernel_size=3, strides=1, padding='same'):
        super(ConvBnRelu, self).__init__()
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(channel, kernel_size, strides=strides, padding=padding),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

    def call(self, x, training=None):
        y = self.model(x, training=training)
        return y


class InceptionBlock(tf.keras.Model):
    def __init__(self, channel, strides=1):
        super(InceptionBlock, self).__init__()
        self.channel = channel
        self.strides = strides

        self.conv1 = ConvBnRelu(self.channel, kernel_size=3, strides=self.strides)
        self.conv2 = ConvBnRelu(self.channel, kernel_size=3, strides=self.strides)
        self.conv3_1 = ConvBnRelu(self.channel, kernel_size=3, strides=self.strides)
        self.conv3_2 = ConvBnRelu(self.channel, kernel_size=3, strides=1)

        self.pool = tf.keras.layers.MaxPooling2D(3, strides=1, padding='same')
        self.pool_conv = ConvBnRelu(self.channel, kernel_size=3, strides=self.strides)

    def call(self, x, training=None):
        x1 = self.conv1(x, training=training)

        x2 = self.conv2(x, training=training)

        x3_1 = self.conv3_1(x, training=training)
        x3_2 = self.conv3_2(x3_1, training=training)

        x4 = self.pool(x)
        x4 = self.pool_conv(x4)

        # import pdb;
        # pdb.set_trace()

        y = tf.concat([x1, x2, x3_2, x4], axis=3)

        return y


class Inception(tf.keras.Model):
    def __init__(self, num_layers, num_classes, init_channel=16, **kwargs):
        super(Inception, self).__init__()

        self.in_channels = init_channel
        self.out_channels = init_channel
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.conv1 = ConvBnRelu(init_channel)
        self.blocks = tf.keras.models.Sequential(name='dynamic-blocks')

        for block_id in range(self.num_layers):
            for layer_id in range(2):
                if layer_id == 0:
                    block = InceptionBlock(self.out_channels, strides=2)
                else:
                    block = InceptionBlock(self.out_channels, strides=1)

                self.blocks.add(block)

            self.out_channels = self.out_channels * 2

        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(num_classes)

    def call(self, x, training=None):
        y = self.conv1(x, training=training)
        y = self.blocks(y, training=training)
        y = self.avg_pool(y)
        y = self.fc(y)

        return y
