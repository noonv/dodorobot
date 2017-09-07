#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
based on 

Автоэнкодеры в Keras, Часть 5: GAN(Generative Adversarial Networks) и tensorflow  https://habrahabr.ru/post/332000/

http://www.rricard.me/machine/learning/generative/adversarial/networks/keras/tensorflow/2017/04/05/gans-part2.html

https://oshearesearch.com/index.php/2016/07/01/mnist-generative-adversarial-model-in-keras/
	https://github.com/osh/KerasGAN/blob/master/mnist_gan.py
	
	
Deep Convolutional GANs (DCGAN):
Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks  https://arxiv.org/abs/1511.06434

https://github.com/kyloon/dcgan

'''

__author__ = 'noonv'

import numpy as np
np.random.seed(42)

import pandas as pd

from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Reshape, Dropout
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Conv2DTranspose
from keras.layers import GlobalAveragePooling2D
from keras.layers import concatenate
from keras.layers import Lambda
from keras.layers import RepeatVector
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.objectives import binary_crossentropy

from keras.layers.advanced_activations import LeakyReLU

from keras.optimizers import Adam, RMSprop, SGD

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K
#For 2D data (e.g. image), "channels_last" assumes (rows, cols, channels) while "channels_first" assumes  (channels, rows, cols).
K.set_image_data_format('channels_first')

from theano.version import version as theano_version
from keras import __version__ as keras_version

#%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt

import cv2
import os
import sys

from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle

# for GPU
from theano import config
import theano.sandbox.cuda
config.floatX = 'float32'
print(config.floatX)
theano.sandbox.cuda.use("gpu0")

import load_data
import prepare_images
import rotate_image

pizza_eng_names, pizza_imgs = prepare_images.load_photos()

channels, height, width = 3, 32, 32
batch_size = 16

image_list = load_data.resize_rotate_flip(pizza_imgs[0], (height, width))

print(len(image_list))

images_count = len(image_list)

image_list = np.array(image_list, dtype=np.float32)
image_list = image_list.transpose((0, 3, 1, 2))
#image_list /= 255.0
image_list -= 127.5
image_list /= 127.5
print(image_list.shape)

x_train = image_list[:600]
x_test = image_list[600:]

print(x_train.shape, x_test.shape)

latent_dim = 512

def make_trainable(net, val):
	net.trainable = val
	for l in net.layers:
		l.trainable = val

def create_gan(channels, height, width):
	
	input_img = Input(shape=(channels, height, width))
	
	m_height, m_width = int(height/8), int(width/8)
	
	# generator
	z = Input(shape=(latent_dim, ))
	x = Dense(256*m_height*m_width)(z)
	#x = BatchNormalization()(x)
	x = Activation('relu')(x)
	#x = Dropout(0.3)(x)
	
	x = Reshape((256, m_height, m_width))(x)

	x = Conv2DTranspose(256, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu')(x)
	
	x = Conv2DTranspose(128, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu')(x)
	
	x = Conv2DTranspose(64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu')(x)
	
	x = Conv2D(channels, (5, 5), padding='same')(x)
	g = Activation('tanh')(x)
	
	generator = Model(z, g, name='Generator')
	
	# discriminator
	x = Conv2D(128, (5, 5), padding='same')(input_img)
	#x = BatchNormalization()(x)
	x = LeakyReLU()(x)
	#x = Dropout(0.3)(x)
	x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
	x = Conv2D(256, (5, 5), padding='same')(x)
	x = LeakyReLU()(x)
	x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
	x = Conv2D(512, (5, 5), padding='same')(x)
	x = LeakyReLU()(x)
	x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
	x = Flatten()(x)
	x = Dense(2048)(x)
	x = LeakyReLU()(x)
	x = Dense(1)(x)
	d = Activation('sigmoid')(x)
	
	discriminator = Model(input_img, d, name='Discriminator')
	
	gan = Sequential()
	gan.add(generator)
	make_trainable(discriminator, False) #discriminator.trainable = False
	gan.add(discriminator)
	
	return generator, discriminator, gan


def get_image_from_net_data(data):
	res = data.transpose((1, 2, 0))
	#res *= 255.0
	res *= 127.5
	res += 127.5
	res = np.array(res, dtype=np.uint8)
	return res


gan_gen, gan_ds, gan = create_gan(channels, height, width)

gan_gen.summary()
gan_ds.summary()
gan.summary()

opt = Adam(lr=1e-3)
gopt = Adam(lr=1e-4)
dopt = Adam(lr=1e-4)

gan_gen.compile(loss='binary_crossentropy', optimizer=gopt)
gan.compile(loss='binary_crossentropy', optimizer=opt)

make_trainable(gan_ds, True)
gan_ds.compile(loss='binary_crossentropy', optimizer=dopt)

def show_gan_loss(d_loss, g_loss):
	print('Show history...')
	plt.figure()
	plt.plot(d_loss)
	plt.plot(g_loss)
	plt.title('GAN loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['discriminator', 'GAN'], loc='best')
	plt.show()

def save_images(generated_images, dst='temp/gen'):
	image = np.zeros((height, width*batch_size, channels), dtype=generated_images.dtype)
	for index, img in enumerate(generated_images):
		image[0:height, index*width:(index+1)*width] = get_image_from_net_data(img)
	cv2.imwrite(dst+'_'+str(epoch)+'.png', image)

d_loss = []
g_loss = []

# load weights
if os.path.exists('gan_gen_weights_'+str(height)+'.h5'):
	print('Load generator weights...')
	gan_gen.load_weights('gan_gen_weights_'+str(height)+'.h5')
if os.path.exists('gan_ds_weights_'+str(height)+'.h5'):
	print('Load discriminator weights...')
	gan_ds.load_weights('gan_ds_weights_'+str(height)+'.h5')

epochs = 50000

for epoch in range(epochs):
	print('Epoch {} from {} ...'.format(epoch, epochs))
	
	n = x_train.shape[0]
	image_batch = x_train[np.random.randint(0, n, size=batch_size),:,:,:]    
	
	noise_gen = np.random.uniform(-1, 1, size=[batch_size, latent_dim])

	generated_images = gan_gen.predict(noise_gen, batch_size=batch_size)
	
	if epoch % 10 == 0:
		print('Save gens ...')
		save_images(generated_images)
		gan_gen.save_weights('temp/gan_gen_weights_'+str(height)+'.h5', True)
		gan_ds.save_weights('temp/gan_ds_weights_'+str(height)+'.h5', True)
		# save loss
		df = pd.DataFrame( {'d_loss': d_loss, 'g_loss': g_loss} )
		df.to_csv('temp/gan_loss.csv', index=False)
	
	x_train2 = np.concatenate( (image_batch, generated_images) )
	y_tr2 = np.zeros( [2*batch_size, 1] )
	y_tr2[:batch_size] = 1
	
	d_history = gan_ds.train_on_batch(x_train2, y_tr2)
	print('d:', d_history)
	d_loss.append( d_history )

	noise_gen = np.random.uniform(-1, 1, size=[batch_size, latent_dim])
	g_history = gan.train_on_batch(noise_gen, np.ones([batch_size, 1]))
	print('g:', g_history)
	g_loss.append( g_history )

gan_gen.save_weights('gan_gen_weights_'+str(height)+'.h5')
gan_ds.save_weights('gan_ds_weights_'+str(height)+'.h5')

show_gan_loss(d_loss, g_loss)

n = batch_size
noise = np.random.uniform(-1, 1, size = [n, latent_dim])
print(noise.shape)
generated_images = gan_gen.predict(noise, batch_size=batch_size)
for i in range(n):
	image = get_image_from_net_data(generated_images[i])
	cv2.imshow('generated_images', image)
	train = get_image_from_net_data(x_test[i])
	cv2.imshow('train', train)
	k = cv2.waitKey(0) & 0xFF
	if k == 27:	# wait for ESC key to exit
		exit()

if __name__ == '__main__':
	print('Start...')
	print('Theano version: {}'.format(theano_version))
	print('Keras version: {}'.format(keras_version))
