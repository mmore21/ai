import tensorflow as tf

class ResNet(tf.keras.Model):
    def __init__(self):
        pass

class ResNet18():
    return ResNet()

if __name__ == "__main__":
    net = ResNet18()
    y = net(tf.random.uniform(shape=[1, 3, 32, 32]))
    print(y.shape())

