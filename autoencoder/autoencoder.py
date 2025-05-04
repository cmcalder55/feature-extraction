#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Created On:    2021/01/01
Last Revision: 2024/09/21

Deep Auto-Encoder.
"""

# imports
import numpy as np
from keras._tf_keras.keras.layers import Input, Dense
from keras._tf_keras.keras.models import Model

# metadata
__author__= "Cameron Calder"
__copyright__ = "(C)Copyright 2024-Present, Cameron Calder"
__license__=""
__version__= "1.0.0"


class ImgAutoEncoder():
    """Deep auto-encoder for images.
    
    Images can be color or grayscale.
    """
    
    def __init__(self, encoding_dim, input_dim=None, n_layers=None):
        self.encoding_dim = encoding_dim
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.compression = None
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        
    def rgb2gray(self, img):
        """Convert image from RGB to greyscale.
        
        Luminosity transform using Rec. 709:
        http://www.glennchan.info/articles/technical/rec709rec601/rec709rec601.html
        """
        
        color_weights = np.array([0.02126, 0.7152, 0.0722])
        grayscale = np.dot(img[...,:3], color_weights)
        return grayscale

    def prep_data(self, x_train, x_test):

        if len(x_train.shape) > 3:
            x_train = np.array([self.rgb2gray(img) for img in x_train])
            x_test = np.array([self.rgb2gray(img) for img in x_test])

        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

        self.input_dim = x_train.shape[1]
        print(f"Train data shape: {x_train.shape}\nTest data shape: {x_test.shape}\n")

        return x_train, x_test

    def _build(self, n_layers, dim_scale=2, activation="relu", final_activation="sigmoid"):

        self.n_layers = n_layers

        if not self.compression:
            self.compression = float(self.input_dim)/self.encoding_dim
            print(f"Compression factor: {self.compression}")

        # input placeholder
        input_img = Input(shape=(self.input_dim,))
        layer_thickness = range(self.n_layers-1, -1, -1)
        layer_units = []

        encoded = input_img
        # encoded representation of the layer inputs
        for t in layer_thickness:
            units = (dim_scale**t)*self.encoding_dim
            if units > self.input_dim:
                print(f"Encoding dims [{units}] too large")
                return False, False, False

            layer_units.append(units)
            encoded = Dense(units, activation=activation)(encoded)

        # lossy reconstruction of the layer inputs
        decoded = encoded
        for units in reversed(layer_units[:-1]):
            decoded = Dense(units, activation=activation)(decoded)
        # final decoder layer has the same size as initial layer, sigmoid activation
        decoded = Dense(self.input_dim, activation=final_activation)(decoded)

        # this model maps an input to its reconstruction
        autoencoder = Model(input_img, decoded, name=f"AutoEncoder_{self.n_layers}Layer")
        autoencoder.summary()
        self.autoencoder = autoencoder

        # encoded input
        encoder = Model(input_img, encoded)
        encoded_input = Input(shape=(self.encoding_dim,))
        self.encoder = encoder

        # dynamically construct decoded input
        decoded_input = encoded_input
        
        # Iterate over the autoencoder's layers to construct the decoder dynamically
        for layer in autoencoder.layers[-self.n_layers:]:
            decoded_input = layer(decoded_input)

        decoder = Model(encoded_input, decoded_input)
        self.decoder = decoder

    def _fit(self, x_test, x_train, epochs=20, batch_size=256, optimizer="adam", loss="binary_crossentropy"):
        
        self.autoencoder.compile(optimizer=optimizer, loss=loss)
        self.autoencoder.fit(x_train, x_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(x_test, x_test))

        # encode and decode some digits from the *test* set
        encoded_imgs = self.encoder.predict(x_test)
        decoded_imgs = self.decoder.predict(encoded_imgs)

        return encoded_imgs, decoded_imgs
