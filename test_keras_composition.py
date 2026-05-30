import keras
import numpy as np

class QuantVars:
    def __init__(self, layer, name):
        self.scale = layer.add_weight(
            name=f"{name}_scale", shape=(), initializer="ones", trainable=True
        )

class MyLayer(keras.Layer):
    def build(self, input_shape):
        self.quant = QuantVars(self, "input")
        super().build(input_shape)
        
    def call(self, x):
        return x * self.quant.scale

layer = MyLayer()
layer.build((None, 10))
print("Trainable weights:", [w.name for w in layer.trainable_weights])
