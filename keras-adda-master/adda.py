from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, clone_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras import regularizers
from keras.utils import np_utils
import keras.backend as K
import keras

import tensorflow as tf
from datasets import get_dataset

import matplotlib.pyplot as plt

import sys
import os
import numpy as np
import argparse

class ADDA():
    def __init__(self, lr):
        # Input shape
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        self.src_flag = False
        self.disc_flag = False
        
        self.discriminator_decay_rate = 3 #iterations
        self.discriminator_decay_factor = 0.5
        self.src_optimizer = Adam(lr, 0.5)
        self.tgt_optimizer = Adam(lr, 0.5)
        
    def define_source_encoder(self, weights=None):
    
        #self.source_encoder = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=self.img_shape, pooling=None, classes=10)
        
        self.source_encoder = Sequential()
        inp = Input(shape=self.img_shape)
        x = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.img_shape, padding='same')(inp)
        x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=self.img_shape, padding='same')(inp)
        x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        self.source_encoder = Model(inputs=(inp), outputs=(x))
        
        self.src_flag = True
        
        if weights is not None:
            self.source_encoder.load_weights(weights, by_name=True)
    
    def define_target_encoder(self, weights=None):
        
        if not self.src_flag:
            self.define_source_encoder()
        
        with tf.device('/gpu:1'):
            self.target_encoder = clone_model(self.source_encoder)
        
        if weights is not None:
            self.target_encoder.load_weights(weights, by_name=True)
        
    def get_source_classifier(self, model, weights=None):
        
        x = Flatten()(model.output)
        x = Dense(128, activation='relu')(x)
        x = Dense(10, activation='softmax')(x)
        
        source_classifier_model = Model(inputs=(model.input), outputs=(x))
        
        if weights is not None:
            source_classifier_model.load_weights(weights)
        
        return source_classifier_model
    
    def define_discriminator(self, shape):
        
        inp = Input(shape=shape)
        x = Flatten()(inp)
        x = Dense(128, activation=LeakyReLU(alpha=0.3), kernel_regularizer=regularizers.l2(0.01), name='discriminator1')(x)
        
        x = Dense(2, activation='sigmoid', name='discriminator2')(x)
        
        self.disc_flag = True
        self.discriminator_model = Model(inputs=(inp), outputs=(x), name='discriminator')
    
    def tensorboard_log(self, callback, names, logs, batch_no):
        
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, batch_no)
            callback.writer.flush()
    
    def get_discriminator(self, model, weights=None):
        
        if not self.disc_flag:
            self.define_discriminator(model.output_shape[1:])
        
        disc = Model(inputs=(model.input), outputs=(self.discriminator_model(model.output)))
        
        if weights is not None:
            disc.load_weights(weights, by_name=True)
        
        return disc
      
    def train_source_model(self, model, epochs=2000, batch_size=128, save_interval=1, start_epoch=0):

        (train_x, train_y), (test_x, test_y) = get_dataset('svhn')
        
        datagen = ImageDataGenerator(data_format='channels_last', 
                                rescale=1./255, 
                                rotation_range=40, 
                                width_shift_range=0.2, 
                                height_shift_range=0.2)
       
        evalgen = ImageDataGenerator(data_format='channels_last', 
                                rescale=1./255)
        
        model.compile(loss='categorical_crossentropy', optimizer=self.src_optimizer, metrics=['accuracy'])
        
        if not os.path.isdir('data'):
            os.mkdir('data')
        
        saver = keras.callbacks.ModelCheckpoint('data/svhn_encoder_{epoch:02d}.hdf5', 
                                        monitor='val_loss', 
                                        verbose=1, 
                                        save_best_only=False, 
                                        save_weights_only=True, 
                                        mode='auto', 
                                        period=save_interval)

        scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=10, verbose=0, mode='min')

        if not os.path.isdir('data/tensorboard'):
            os.mkdir('data/tensorboard')
    
        visualizer = keras.callbacks.TensorBoard(log_dir=os.path.join('data/tensorboard'), 
                                            histogram_freq=0, 
                                            write_graph=True, 
                                            write_images=False)
        
        model.fit_generator(datagen.flow(train_x, train_y, batch_size=batch_size, shuffle=True),
                            steps_per_epoch=2000, 
                            epochs=epochs,
                            callbacks=[saver, scheduler, visualizer], 
                            validation_data=evalgen.flow(test_x, test_y, batch_size=batch_size), 
                            initial_epoch=start_epoch)
        
    def train_target_discriminator(self, source_model=None, src_discriminator=None, tgt_discriminator=None, epochs=2000, batch_size=100, save_interval=1, start_epoch=0, num_batches=100):   
    
        (source_x, _), (_,_) = get_dataset('svhn')
        
        src_datagen = ImageDataGenerator(data_format='channels_last', 
                                rescale=1./255, 
                                rotation_range=40, 
                                width_shift_range=0.2, 
                                height_shift_range=0.2)
        
        (target_x, _), (_,_) = get_dataset('mnist')
        
        tgt_datagen = ImageDataGenerator(data_format='channels_last', 
                                rescale=1./255, 
                                rotation_range=40, 
                                width_shift_range=0.2, 
                                height_shift_range=0.2)
        
        self.define_source_encoder(source_model)
                
        for layer in self.source_encoder.layers:
            layer.trainable = False
        
        source_discriminator = self.get_discriminator(self.source_encoder, src_discriminator)
        target_discriminator = self.get_discriminator(self.target_encoder, tgt_discriminator)
        
        if src_discriminator is not None:
            source_discriminator.load_weights(src_discriminator)
        if tgt_discriminator is not None:
            target_discriminator.load_weights(tgt_discriminator)
        
        source_discriminator.compile(loss = "binary_crossentropy", optimizer=self.tgt_optimizer, metrics=['accuracy'])
        target_discriminator.compile(loss = "binary_crossentropy", optimizer=self.tgt_optimizer, metrics=['accuracy'])
        
        callback1 = keras.callbacks.TensorBoard('data/tensorboard')
        callback1.set_model(source_discriminator)
        callback2 = keras.callbacks.TensorBoard('data/tensorboard')
        callback2.set_model(target_discriminator)
        src_names = ['src_discriminator_loss', 'src_discriminator_acc']
        tgt_names = ['tgt_discriminator_loss', 'tgt_discriminator_acc']
        
        for iteration in range(start_epoch, epochs):
            
            avg_loss, avg_acc, index = [0, 0], [0, 0], 0
        
            for mnist,svhn in zip(src_datagen.flow(source_x, None, batch_size=batch_size), tgt_datagen.flow(target_x, None, batch_size=batch_size)):
                l1, acc1 = source_discriminator.train_on_batch(svhn, np_utils.to_categorical(np.zeros(svhn.shape[0]), 2))
                l2, acc2 = target_discriminator.train_on_batch(mnist, np_utils.to_categorical(np.ones(mnist.shape[0]), 2))
                index+=1
                loss, acc = (l1+l2)/2, (acc1+acc2)/2
                print (iteration+1,': ', index,'/', num_batches, '; Loss: %.4f'%loss, ' (', '%.4f'%l1, '%.4f'%l2, '); Accuracy: ', acc, ' (', '%.4f'%acc1, '%.4f'%acc2, ')')
                avg_loss[0] += l1
                avg_acc[0] += acc1
                avg_loss[1] += l2
                avg_acc[1] += acc2
                if index%num_batches == 0:
                    break
            
            if iteration%self.discriminator_decay_rate==0:
                lr = K.get_value(source_discriminator.optimizer.lr)
                K.set_value(source_discriminator.optimizer.lr, lr*self.discriminator_decay_factor)
                lr = K.get_value(target_discriminator.optimizer.lr)
                K.set_value(target_discriminator.optimizer.lr, lr*self.discriminator_decay_factor)
                print ('Learning Rate Decayed to: ', K.get_value(target_discriminator.optimizer.lr))
            
            if iteration%save_interval==0:
                source_discriminator.save_weights('data/discriminator_svhn_ %02d.hdf5'%iteration)
                target_discriminator.save_weights('data/discriminator_mnist_%02d.hdf5'%iteration)
                
            self.tensorboard_log(callback1, src_names, [avg_loss[0]/mnist.shape[0], avg_acc[0]/mnist.shape[0]], iteration)
            self.tensorboard_log(callback2, tgt_names, [avg_loss[1]/mnist.shape[0], avg_acc[1]/mnist.shape[0]], iteration)
    
    def eval_source_classifier(self, model, dataset='svhn', batch_size=128, domain='Source'):
        
        (train_x,_), (test_x, test_y) = get_dataset(dataset)
        src_datagen = ImageDataGenerator(data_format='channels_last', 
                                        rescale=1./255)
                      
        model.compile(loss='categorical_crossentropy', optimizer=self.src_optimizer, metrics=['accuracy'])

        scores = model.evaluate_generator(src_datagen.flow(test_x[:10000], test_y[:10000]),10000)
        print('%s %s Classifier Test loss:%.5f'%(dataset.upper(), domain, scores[0]))
        print('%s %s Classifier Test accuracy:%.2f%%'%(dataset.upper(), domain, float(scores[1])*100))            
            
    def eval_target_classifier(self, source_model, target_discriminator, dataset='mnist'):
        
        self.define_target_encoder()
        model = self.get_source_classifier(self.target_encoder, source_model)
        model.load_weights(target_discriminator, by_name=True)
        model.summary()
        self.eval_source_classifier(model, dataset=dataset, domain='Target')
         
if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-s', '--source_weights', required=False, help="Path to weights file to load source model for training classification/adaptation")
    ap.add_argument('-e', '--start_epoch', type=int,default=1, required=False, help="Epoch to begin training source model from")
    ap.add_argument('-n', '--discriminator_epochs', type=int, default=10000, help="Max number of steps to train discriminator")
    ap.add_argument('-l', '--lr', type=float, default=0.0001, help="Initial Learning Rate")
    ap.add_argument('-f', '--train_discriminator', action='store_true', help="Train discriminator model (if TRUE) vs Train source classifier")
    ap.add_argument('-a', '--source_discriminator_weights', help="Path to weights file to load source discriminator")
    ap.add_argument('-b', '--target_discriminator_weights', help="Path to weights file to load target discriminator")
    ap.add_argument('-t', '--eval_source_classifier', default=None, help="Path to source classifier model to test/evaluate")
    ap.add_argument('-d', '--eval_target_classifier', default=None, help="Path to target discriminator model to test/evaluate")
    args = ap.parse_args()
    
    adda = ADDA(args.lr)
    adda.define_source_encoder()
    
    if not args.train_discriminator:
        if args.eval_source_classifier is None:
            model = adda.get_source_classifier(adda.source_encoder, args.source_weights)
            adda.train_source_model(model, start_epoch=args.start_epoch-1) 
        else:
            model = adda.get_source_classifier(adda.source_encoder, args.eval_source_classifier)
            adda.eval_source_classifier(model, 'mnist')
            adda.eval_source_classifier(model, 'svhn')
    adda.define_target_encoder(args.source_weights)
    
    if args.train_discriminator:
        adda.train_target_discriminator(epochs=args.discriminator_epochs, 
                                        source_model=args.source_weights, 
                                        src_discriminator=args.source_discriminator_weights, 
                                        tgt_discriminator=args.target_discriminator_weights,
                                        start_epoch=args.start_epoch-1)
    if args.eval_target_classifier is not None:
        adda.eval_target_classifier(args.eval_source_classifier, args.eval_target_classifier)
    
