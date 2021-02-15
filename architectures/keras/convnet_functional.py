import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

class ConvNet:
    def __init__(self):
        self.model = create_model



    def create_model(self, inputs):

        # --- Define kwargs dictionary
        kwargs = {
            'kernel_size': (1, 3, 3),
            'padding': 'same'
        }

        # --- Define lambda functions
        conv = lambda x, filters, strides : tf.keras.layers.Conv3D(filters=filters, strides=strides, **kwargs)(x)
        tran = lambda x, filters, strides : tf.keras.layers.Conv3DTranspose(filters=filters, strides=strides, **kwargs)(x)
        norm = lambda x : tf.keras.layers.BatchNormalization()(x)
        relu = lambda x : tf.keras.layers.LeakyReLU()(x)

        # --- Define stride-1, stride-2 blocks
        conv1 = lambda filters, x : relu(norm(conv(x, filters, strides=1)))
        conv2 = lambda filters, x : relu(norm(conv(x, filters, strides=(1, 2, 2))))
        tran2 = lambda filters, x : relu(norm(tran(x, filters, strides=(1, 2, 2))))

        a = 1

        l1 = conv1(int(a*8), inputs)
        l2 = conv1(int(a*16), conv2(int(a*16), l1))
        l3 = conv1(int(a*32), conv2(int(a*32), l1))
        l4 = conv1(int(a*64), conv2(int(a*64), l1))
        l5 = conv1(int(a*128), conv2(int(a*128), l1))

        model = tf.keras.Model(inputs=inputs, outputs=logits)

        model.compile(
            optimizer = tf.optimizers.Adam(learning_rate=p['lr']),
            loss = {'lbl': tf.losses.sparse_categorical_crossentropy(from_logits=True)},
            metrics = {'lbl': tf.metrics.sparse_categorical_accuracy()},
            experimental_run_tf_function=False
        )

        return model

