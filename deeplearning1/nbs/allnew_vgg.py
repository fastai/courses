import os, json, sys
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.utils.data_utils import get_file
from keras.preprocessing import image
from keras.optimizers import SGD, RMSprop, Adam
from IPython.display import FileLink

vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))
opt = RMSprop(lr=0.00001, rho=0.7)

def vgg_preprocess(x):
    x = x - vgg_mean
    return x[:, ::-1] # reverse axis rgb->bgr

# def proc_wgts(layer): return [o/2 for o in layer.get_weights()]

def proc_wgts(layer, prev_p, new_p):
    scal = (1-prev_p)/(1-new_p)
    return [weight*scal for weight in layer.get_weights()]

class adjusted_vgg():
    """Building a model off of a set of input layers working from there."""
    def __init__(self):
        self.FILE_PATH = "http://www.platform.ai/models/"

    def create_conv_model(self, conv_layers):
        model = self.model = Sequential(conv_layers)

    def create_fc_model(self, train_features, p):
        fc_model = self.fc_model = Sequential()
        fc_model.add(MaxPooling2D(input_shape=train_features.shape[1:]))
        fc_model.add(Flatten())
        fc_model.add(Dense(4096, activation='relu'))
        fc_model.add(Dropout(p))
        fc_model.add(BatchNormalization())
        fc_model.add(Dense(4096, activation='relu'))
        fc_model.add(Dropout(p))
        fc_model.add(BatchNormalization())
        fc_model.add(Dense(2, activation='softmax'))
    
    def add_fc_weights(self, fc_layers, p):
        fc_model = self.fc_model
        for l in fc_model.layers: 
            if type(l)==Dense: l.set_weights(proc_wgts(l, 1-p, p))

    def model_compiler(self, opt, loss='categorical_crossentropy', metrics=['accuracy']):
        fc_model = self.fc_model
        fc_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    def mnn_batches(self, file_path, batch_size, train=True, class_mode='categorical', shuffle=False):
        if train:
            gen = image.ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.15, zoom_range=0.1, channel_shift_range=10., horizontal_flip=True)
        else:
            gen = image.ImageDataGenerator()
        batches = gen.flow_from_directory(directory=file_path, target_size = (224, 224), class_mode=class_mode, batch_size=batch_size, shuffle=shuffle)
        return batches

    def dense_layer_model(self, fc_layers, input_shape):
        fc_model = Sequential()
        fc_model.add(MaxPooling2D(input_shape=input_shape))
        fc_model.add(Flatten())
        # fc_model.add(Dense(4096, activation='relu'))
        # fc_model.add(Dropout(0.))
#        fc_model.add(BatchNormalization())
        fc_model.add(Dense(4096, activation='relu'))
        fc_model.add(Dropout(0.))
#        fc_model.add(BatchNormalization())
        fc_model.add(Dense(2, activation='softmax'))

        for l1,l2 in zip(fc_model.layers, fc_layers): l1.set_weights(proc_wgts(l2))

        fc_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return fc_model
            
    def test(self, test_path, batch_size):
        test_batches = self.get_batches(test_path, shuffle=False, batch_size=batch_size, class_mode=None)
        predictions = self.model.predict_generator(test_batches, test_batches.nb_sample)
        return test_batches, predictions

    def submission_formatter(self, filenames, predictions, pred_column_index, clip_min=0, clip_max=1):
        prediction_values = predictions[:,pred_column_index]
        prediction_values = prediction_values.clip( min=clip_min, max=clip_max )
        ids = np.array([int(f[(f.find('/') + 1):f.find('.')]) for f in filenames])
        subm = np.stack([ids,prediction_values], axis=1)
        return subm

    def csv_creator(self, submission, base_directory, submission_file_name):
        np.savetxt(base_directory+submission_file_name, submission, fmt='%d,%.5f', header='id,label', comments='')
        
        
class preloaded_vgg():
    """Standard implementation of the VGG 16 Imagenet Model"""
    
    def __init__(self):
        self.FILE_PATH = "http://www.platform.ai/models/"
        self.create()
        self.get_classes()

    def get_classes(self):
        fname = 'imagenet_class_index.json'
        fpath = get_file(fname, self.FILE_PATH+fname, cache_subdir='models')
        with open(fpath) as f:
            class_dict = json.load(f)
        self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))]        

    def ConvBlock(self, layers, filters):
        model = self.model
        for i in range(layers):
            model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(filters, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    def FCBlock(self):
        model = self.model
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))

    def create(self):
        model = self.model = Sequential()
        model.add(Lambda(vgg_preprocess, input_shape=(3,224,224)))
        
        self.ConvBlock(2, 64)
        self.ConvBlock(2, 128)
        self.ConvBlock(3, 256)
        self.ConvBlock(3, 512)
        self.ConvBlock(3, 512)
        
        model.add(Flatten())
        self.FCBlock()
        self.FCBlock()
        model.add(Dense(1000, activation='softmax'))
        
        fname = 'vgg16.h5'
        model.load_weights(get_file(fname, self.FILE_PATH+fname, cache_subdir='models'))
        
    def mnn_finetuner(self):
        model = self.model
        print 'starting number of layers: ', len(model.layers)
        model.pop()
        for layer in model.layers: layer.trainable=False
        model.add(Dense(2, activation='softmax'))
        print 'ending number of layers: ', len(model.layers)
    
    def layer_divider(self):
        last_conv_idx = [index for index,layer in enumerate(self.model.layers) if type(layer) is Convolution2D][-1]
        conv_layers = self.model.layers[:last_conv_idx+1]
        fc_layers = self.model.layers[last_conv_idx+1:]
        return conv_layers, fc_layers        
    
    def mnn_batches(self, file_path, train=True, class_mode='categorical', batch_size=8, shuffle=True):
        if train:
            gen = image.ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.15, zoom_range=0.1, channel_shift_range=10., horizontal_flip=True)
        else:
            gen = image.ImageDataGenerator()
        batches = gen.flow_from_directory(directory=file_path, target_size = (224, 224), class_mode=class_mode, batch_size=batch_size, shuffle=shuffle)
        return batches
        
    def fit(self, train_batches, valid_batches, nb_epoch=1):
        self.model.fit_generator(train_batches, samples_per_epoch=train_batches.nb_sample, nb_epoch=nb_epoch, validation_data=valid_batches, nb_val_samples=valid_batches.nb_sample)
        
    def test(self, test_path, batch_size):
        test_batches = self.get_batches(test_path, shuffle=False, batch_size=batch_size, class_mode=None)
        predictions = self.model.predict_generator(test_batches, test_batches.nb_sample)
        return test_batches, predictions

    def submission_formatter(self, filenames, predictions, pred_column_index, clip_min=0, clip_max=1):
        prediction_values = predictions[:,pred_column_index]
        prediction_values = prediction_values.clip( min=clip_min, max=clip_max )
        ids = np.array([int(f[(f.find('/') + 1):f.find('.')]) for f in filenames])
        subm = np.stack([ids,prediction_values], axis=1)
        return subm

    def csv_creator(self, submission, base_directory, submission_file_name):
        np.savetxt(base_directory+submission_file_name, submission, fmt='%d,%.5f', header='id,label', comments='')

        
## Things to add to improve model:
# data augmentation. only for training data, in the batches_phase
# pop off models until the convolutional block.
