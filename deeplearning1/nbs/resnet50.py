from __future__ import division, print_function

import os, json
from glob import glob
import numpy as np
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers import Input, Activation, merge
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
import keras.preprocessing.image as image
from keras.utils.data_utils import get_file
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.applications.resnet50 import identity_block, conv_block


class Resnet50():
    """The Resnet 50 Imagenet model"""


    def __init__(self, size=(224,224), include_top=True):
        self.FILE_PATH = 'http://files.fast.ai/models/'
        self.vgg_mean = np.array([123.68, 116.779, 103.939]).reshape((3,1,1))
        self.create(size, include_top)
        self.get_classes()


    def get_classes(self):
        fname = 'imagenet_class_index.json'
        fpath = get_file(fname, self.FILE_PATH+fname, cache_subdir='models')
        with open(fpath) as f:
            class_dict = json.load(f)
        self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))]

    def predict(self, imgs, details=False):
        all_preds = self.model.predict(imgs)
        idxs = np.argmax(all_preds, axis=1)
        preds = [all_preds[i, idxs[i]] for i in range(len(idxs))]
        classes = [self.classes[idx] for idx in idxs]
        return np.array(preds), idxs, classes


    def vgg_preprocess(self, x):
        x = x - self.vgg_mean
        return x[:, ::-1] # reverse axis bgr->rgb


    def create(self, size, include_top):
        input_shape = (3,)+size
        img_input = Input(shape=input_shape)
        bn_axis = 1

        x = Lambda(self.vgg_preprocess)(img_input)
        x = ZeroPadding2D((3, 3))(x)
        x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        for n in ['b','c','d']: x = identity_block(x, 3, [128, 128, 512], stage=3, block=n)
        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        for n in ['b','c','d', 'e', 'f']: x = identity_block(x, 3, [256, 256, 1024], stage=4, block=n)

        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        if include_top:
            x = AveragePooling2D((7, 7), name='avg_pool')(x)
            x = Flatten()(x)
            x = Dense(1000, activation='softmax', name='fc1000')(x)
            fname = 'resnet50.h5'
        else:
            fname = 'resnet_nt.h5'

        self.img_input = img_input
        self.model = Model(self.img_input, x)
        convert_all_kernels_in_model(self.model)
        self.model.load_weights(get_file(fname, self.FILE_PATH+fname, cache_subdir='models'))


    def get_batches(self, path, gen=image.ImageDataGenerator(),class_mode='categorical', shuffle=True, batch_size=8):
        return gen.flow_from_directory(path, target_size=(224,224),
                class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)


    def finetune(self, batches):
        model = self.model
        model.layers.pop()
        for layer in model.layers: layer.trainable=False
        m = Dense(batches.nb_class, activation='softmax')(model.layers[-1].output)
        self.model = Model(model.input, m)
        self.model.compile(optimizer=RMSprop(lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])


    def fit(self, batches, val_batches, nb_epoch=1):
        self.model.fit_generator(batches, samples_per_epoch=batches.nb_sample, nb_epoch=nb_epoch,
                validation_data=val_batches, nb_val_samples=val_batches.nb_sample)

    def test(self, path, batch_size=8):
        test_batches = self.get_batches(path, shuffle=False, batch_size=batch_size, class_mode=None)
        return test_batches, self.model.predict_generator(test_batches, test_batches.nb_sample)
