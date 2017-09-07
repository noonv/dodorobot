#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
based on 
Автоэнкодеры в Keras, Часть 4: Conditional VAE	https://habrahabr.ru/post/331664/

https://blog.keras.io/building-autoencoders-in-keras.html
	https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py

'''

__author__ = 'noonv'

import numpy as np
np.random.seed(42)

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import concatenate
from keras.layers import Lambda
from keras.objectives import binary_crossentropy

from keras.optimizers import Adam, RMSprop

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
batch_size = 20

labels, onehotencoder = load_data.load_data()
labels_list = []
j = 0

image_list = []
for pizza_img in pizza_imgs:
	lst = load_data.resize_rotate_flip(pizza_img, (height, width))
	print(len(lst))
	image_list.extend(lst)
	
	lbls = []
	for i in range(len(lst)):
		lbls.append( shuffle(labels[j], random_state=i) )
	labels_list.extend(lbls)
	j += 1

print(len(image_list))

images_count = len(image_list)

image_list = np.array(image_list, dtype=np.float32)
image_list = image_list.transpose((0, 3, 1, 2))
#image_list /= 255.0
image_list -= 127.5
image_list /= 127.5
print(image_list.shape)

labels_list = np.array(labels_list, dtype=np.float32)
print(labels_list.shape)

X, y = shuffle(image_list, labels_list, random_state=0)

print('Split traint and test...')
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=13)
print(x_train.shape, x_test.shape)

	
def create_conv_cvae(channels, height, width, code_h, code_w):
	input_img = Input(shape=(channels, height, width))
	
	input_code = Input(shape=(code_h, code_w))
	flatten_code = Flatten()(input_code)
	
	latent_dim = 512
	m_height, m_width = int(height/4), int(width/4)
	
	x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
	x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
	x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
	flatten_img_features = Flatten()(x)
	x = concatenate([flatten_img_features, flatten_code])
	x = Dense(1024, activation='relu')(x)
	z_mean = Dense(latent_dim)(x)
	z_log_var = Dense(latent_dim)(x)
	
	def sampling(args):
		z_mean, z_log_var = args
		epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)
		return z_mean + K.exp(z_log_var / 2) * epsilon
	l = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

	models = {}
	
	models["encoder"]  = Model([input_img, input_code], l, 'Encoder') 
	models["z_meaner"] = Model([input_img, input_code], z_mean, 'Enc_z_mean')
	models["z_lvarer"] = Model([input_img, input_code], z_log_var, 'Enc_z_log_var')
	
	z = Input(shape=(latent_dim, ))
	input_code_d = Input(shape=(code_h, code_w))
	flatten_code_d = Flatten()(input_code_d)
	x = concatenate([z, flatten_code_d])
	x = Dense(1024)(x)
	x = Dense(16*m_height*m_width)(x)
	x = Reshape((16, m_height, m_width))(x)
	x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	decoded = Conv2D(channels, (3, 3), activation='sigmoid', padding='same')(x)

	models["decoder"] = Model([z, input_code_d], decoded, name='Decoder')
	models["cvae"]	  = Model([input_img, input_code, input_code_d], 
								models["decoder"]([models["encoder"]([input_img, input_code]), input_code_d]), 
								name="CVAE")
	models["style_t"] = Model([input_img, input_code, input_code_d], 
								models["decoder"]([models["z_meaner"]([input_img, input_code]), input_code_d]), 
								name="style_transfer")
	
	
	def vae_loss(x, decoded):
		x = K.reshape(x, shape=(batch_size, channels*height*width))
		decoded = K.reshape(decoded, shape=(batch_size, channels*height*width))
		xent_loss = channels*height*width*binary_crossentropy(x, decoded)
		kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
		return (xent_loss + kl_loss)/2/3/height/width

	return models, vae_loss


models, vae_loss = create_conv_cvae(channels, height, width, 9, 29)
cvae = models["cvae"]

cvae.summary()

cvae.compile(optimizer=Adam(0.001), loss=vae_loss)

callbacks = [
		EarlyStopping(monitor='val_loss', patience=3, verbose=0),
		#ModelCheckpoint(filepath="cnn_reg_checkpoint_weights{val_loss:.2f}.h5", verbose=0, save_best_only=True)
]

#cvae.load_weights('cvae_all_weights_'+str(height)+'.h5')

cvae.fit([x_train, y_train, y_train], x_train, shuffle=True, epochs=40,
		 batch_size=batch_size,
		 validation_data=([x_test, y_test, y_test], x_test),
		 callbacks=callbacks,
		 verbose=1)
		 
cvae.save_weights('cvae_all_weights_'+str(height)+'.h5')

#cvae.load_weights('cvae_all_weights_'+str(height)+'.h5')

n = batch_size
imgs = x_test[:n]
imgs_lbls = y_test[:n]

#decoded_imgs = cvae.predict([imgs, imgs_lbls, imgs_lbls], batch_size=batch_size)

def get_image_from_net_data(data):
	res = data.transpose((1, 2, 0))
	#res *= 255.0
	res *= 127.5
	res += 127.5
	res = np.array(res, dtype=np.uint8)
	return res

def save_images(generated_images, dst='temp/cvae', comment=''):
	image = np.zeros((height, width*batch_size, channels), dtype=generated_images.dtype)
	for index, img in enumerate(generated_images):
		image[0:height, index*width:(index+1)*width] = get_image_from_net_data(img)
	cv2.imwrite(dst+comment+'.png', image)

# test on original images
orig_images = []
for pizza_img in pizza_imgs:
	img = cv2.resize(pizza_img, (height, width))
	orig_images.append(img)

orig_images = np.array(orig_images, dtype=np.float32)
orig_images = orig_images.transpose((0, 3, 1, 2))
#image_list /= 255.0
orig_images -= 127.5
orig_images /= 127.5
print(orig_images.shape)

orig_labels = np.array(labels, dtype=np.float32)
print(orig_labels.shape)

imgs =  orig_images
decoded_imgs = cvae.predict([imgs, orig_labels, orig_labels], batch_size=batch_size)

save_images(np.copy(orig_images), comment='_orig')
save_images(decoded_imgs, comment='_decoded')

stt = models["style_t"]

i = 0
for label in labels:
	i += 1
	lbls = []
	for j in range(batch_size):
		lbls.append(label)
	lbls = np.array(lbls, dtype=np.float32)
	print(i, lbls.shape)
	
	stt_imgs = stt.predict([orig_images, orig_labels, lbls], batch_size=batch_size)
	save_images(stt_imgs, dst='temp/cvae_stt', comment='_'+str(i))


if __name__ == '__main__':
	print('Start...')
	print('Theano version: {}'.format(theano_version))
	print('Keras version: {}'.format(keras_version))
