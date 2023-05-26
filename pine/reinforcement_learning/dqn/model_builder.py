import tensorflow as tf
import keras


# Neural network builder for building QNetworks
class QNetworkBuilder:
    def __init__(self, input_shape, n_outputs, learning_rate):
        # the shape of the input data coming into the network
        self.input_shape = input_shape
        # spcifiy the number of outputs ie the number of actions
        self.n_outputs = n_outputs
        # specify the learning rate of the model
        self.learning_rate = learning_rate
    
    def build(self):
        # create sequential convolutional network
        self.model = keras.models.Sequential(
            [tf.keras.layers.Conv2D(filters=32, kernel_size=(8, 8), activation="relu",strides=(4, 4), name="conv1"),
            tf.keras.layers.Dropout(rate=0.2, name="drop1"),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), activation="relu",strides=(2, 2), name="conv2"),
            tf.keras.layers.Dropout(rate=0.2, name="drop2"),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), name="conv3"),
            tf.keras.layers.BatchNormalization(name="batchnorm"),
            tf.keras.layers.Flatten(name="flatten"),
            tf.keras.layers.Dense(512, activation="relu", name="fc1"),
            tf.keras.layers.Dense(self.n_outputs, name="values")],
        )
        
        # build the model
        input_shape = [None] + list(self.input_shape) # create input shape with the first index being 
                                                      # None so it can take any amount of input data simulatanously
        self.model.build(input_shape=tuple(input_shape))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        self.model.summary()
        return self.model