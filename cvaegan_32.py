#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
based on 
Автоэнкодеры в Keras, часть 6: VAE + GAN  https://habrahabr.ru/post/332074/

https://github.com/tatsy/keras-generative/blob/master/models/cvaegan.py

'''

__author__ = 'noonv'

import numpy as np
np.random.seed(42)

import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import concatenate
from keras.layers import Lambda
from keras.objectives import binary_crossentropy

from keras.layers import GlobalAveragePooling2D
from keras.layers import RepeatVector

from keras.layers import Conv2DTranspose
from keras.layers import BatchNormalization
from keras.layers import Activation

from keras.layers.advanced_activations import LeakyReLU

from keras.optimizers import Adam, RMSprop

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.engine.topology import Layer
from keras.layers import Concatenate
from keras.layers import ELU

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
batch_size = 20 #32

m_height, m_width = int(height/4), int(width/4)

labels, onehotencoder = load_data.load_data()

print('Labels count:', len(labels))
print('Labels shape:', labels[0].shape)
code_h, code_w = labels[0].shape

j = 0
labels_list = []
flatlabels_list = []
image_list = []

print('Resize and rotate images...')
for pizza_img in pizza_imgs:
	lst = load_data.resize_rotate_flip(pizza_img, (height, width))
	print(len(lst))
	image_list.extend(lst)
	
	lbls = []
	flbls = []
	for i in range(len(lst)):
		#lbls.append( shuffle(labels[j], random_state=i) )
		lbls.append( labels[j] )
		flbls.append( labels[j].flatten() )
	labels_list.extend(lbls)
	flatlabels_list.extend(flbls)
	
	j += 1

images_count = len(image_list)
print('Total images count:', images_count)

print('Flat labels:', len(flatlabels_list), flatlabels_list[0].shape)
print(flatlabels_list[0])

image_list = np.array(image_list, dtype=np.float32)
image_list = image_list.transpose((0, 3, 1, 2))
#image_list /= 255.0
image_list -= 127.5
image_list /= 127.5
print(image_list.shape)

labels_list = np.array(labels_list, dtype=np.float32)
print(labels_list.shape)

flatlabels_list = np.array(flatlabels_list, dtype=np.float32)
print(flatlabels_list.shape)

#X, y = shuffle(image_list, labels_list, random_state=0)
X, y = shuffle(image_list, flatlabels_list, random_state=0)

print('Split traint and test...')
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=13)
print(x_train.shape, x_test.shape)

def add_units_to_conv2d(conv2, units):
	dim1 = K.int_shape(conv2)[2]
	dim2 = K.int_shape(conv2)[3]
	dimc = K.int_shape(units)[1]
	repeat_n = dim1*dim2
	count = int( dim1*dim2 / dimc)
	units_repeat = RepeatVector(count+1)(units)
	#print('K.int_shape(units_repeat): ', K.int_shape(units_repeat))
	units_repeat = Flatten()(units_repeat)
	# cut only needed lehgth of code
	units_repeat = Lambda(lambda x: x[:,:dim1*dim2], output_shape=(dim1*dim2,))(units_repeat)
	units_repeat = Reshape((1, dim1, dim2))(units_repeat)
	return concatenate([conv2, units_repeat], axis=1)

def sample_normal(args):
	z_avg, z_log_var = args
	batch_size = K.shape(z_avg)[0]
	z_dims = K.shape(z_avg)[1]
	eps = K.random_normal(shape=(batch_size, z_dims), mean=0.0, stddev=1.0)
	return z_avg + K.exp(z_log_var / 2.0) * eps

def zero_loss(y_true, y_pred):
	return K.zeros_like(y_true)

def set_trainable(model, train):
    """
    Enable or disable training for the model
    """
    model.trainable = train
    for l in model.layers:
        l.trainable = train

class ClassifierLossLayer(Layer):
	__name__ = 'classifier_loss_layer'

	def __init__(self, **kwargs):
		self.is_placeholder = True
		super(ClassifierLossLayer, self).__init__(**kwargs)

	def lossfun(self, c_true, c_pred):
		return K.mean(keras.metrics.categorical_crossentropy(c_true, c_pred))

	def call(self, inputs):
		c_true = inputs[0]
		c_pred = inputs[1]
		loss = self.lossfun(c_true, c_pred)
		self.add_loss(loss, inputs=inputs)

		return c_true

class DiscriminatorLossLayer(Layer):
	__name__ = 'discriminator_loss_layer'

	def __init__(self, **kwargs):
		self.is_placeholder = True
		super(DiscriminatorLossLayer, self).__init__(**kwargs)

	def lossfun(self, y_real, y_fake_f, y_fake_p):
		y_pos = K.ones_like(y_real)
		y_neg = K.zeros_like(y_real)
		loss_real = keras.metrics.binary_crossentropy(y_pos, y_real)
		loss_fake_f = keras.metrics.binary_crossentropy(y_neg, y_fake_f)
		loss_fake_p = keras.metrics.binary_crossentropy(y_neg, y_fake_p)
		return K.mean(loss_real + loss_fake_f + loss_fake_p)

	def call(self, inputs):
		y_real = inputs[0]
		y_fake_f = inputs[1]
		y_fake_p = inputs[2]
		loss = self.lossfun(y_real, y_fake_f, y_fake_p)
		self.add_loss(loss, inputs=inputs)

		return y_real

class GeneratorLossLayer(Layer):
	__name__ = 'generator_loss_layer'

	def __init__(self, **kwargs):
		self.is_placeholder = True
		super(GeneratorLossLayer, self).__init__(**kwargs)

	def lossfun(self, x_r, x_f, f_D_x_f, f_D_x_r, f_C_x_r, f_C_x_f):
		loss_x = K.mean(K.square(x_r - x_f))
		loss_d = K.mean(K.square(f_D_x_r - f_D_x_f))
		loss_c = K.mean(K.square(f_C_x_r - f_C_x_f))

		return loss_x + loss_d + loss_c

	def call(self, inputs):
		x_r = inputs[0]
		x_f = inputs[1]
		f_D_x_r = inputs[2]
		f_D_x_f = inputs[3]
		f_C_x_r = inputs[4]
		f_C_x_f = inputs[5]
		loss = self.lossfun(x_r, x_f, f_D_x_r, f_D_x_f, f_C_x_r, f_C_x_f)
		self.add_loss(loss, inputs=inputs)

		return x_r

class FeatureMatchingLayer(Layer):
	__name__ = 'feature_matching_layer'

	def __init__(self, **kwargs):
		self.is_placeholder = True
		super(FeatureMatchingLayer, self).__init__(**kwargs)

	def lossfun(self, f1, f2):
		f1_avg = K.mean(f1, axis=0)
		f2_avg = K.mean(f2, axis=0)
		return 0.5 * K.mean(K.square(f1_avg - f2_avg))

	def call(self, inputs):
		f1 = inputs[0]
		f2 = inputs[1]
		loss = self.lossfun(f1, f2)
		self.add_loss(loss, inputs=inputs)

		return f1

class KLLossLayer(Layer):
	__name__ = 'kl_loss_layer'

	def __init__(self, **kwargs):
		self.is_placeholder = True
		super(KLLossLayer, self).__init__(**kwargs)

	def lossfun(self, z_avg, z_log_var):
		kl_loss = -0.5 * K.mean(1.0 + z_log_var - K.square(z_avg) - K.exp(z_log_var))
		return kl_loss

	def call(self, inputs):
		z_avg = inputs[0]
		z_log_var = inputs[1]
		loss = self.lossfun(z_avg, z_log_var)
		self.add_loss(loss, inputs=inputs)

		return z_avg

def discriminator_accuracy(x_r, x_f, x_p):
	def accfun(y0, y1):
		x_pos = K.ones_like(x_r)
		x_neg = K.zeros_like(x_r)
		loss_r = K.mean(keras.metrics.binary_accuracy(x_pos, x_r))
		loss_f = K.mean(keras.metrics.binary_accuracy(x_neg, x_f))
		loss_p = K.mean(keras.metrics.binary_accuracy(x_neg, x_p))
		return (1.0 / 3.0) * (loss_r + loss_p + loss_f)

	return accfun

def generator_accuracy(x_p, x_f):
	def accfun(y0, y1):
		x_pos = K.ones_like(x_p)
		loss_p = K.mean(keras.metrics.binary_accuracy(x_pos, x_p))
		loss_f = K.mean(keras.metrics.binary_accuracy(x_pos, x_f))
		return 0.5 * (loss_p + loss_f)

	return accfun

class CVAEGAN():
	def __init__(self,
		input_shape=(channels, height, width),
		num_attrs=code_h*code_w,
		z_dims = 512,
		name='cvaegan',
		**kwargs
	):
		self.input_shape = input_shape
		self.num_attrs = num_attrs
		self.z_dims = z_dims

		self.f_enc = None
		self.f_dec = None
		self.f_dis = None
		self.f_cls = None
		self.enc_trainer = None
		self.dec_trainer = None
		self.dis_trainer = None
		self.cls_trainer = None

		self.build_model()
	
	def train_on_batch(self, x_batch):
		x_r, c = x_batch

		batchsize = len(x_r)
		z_p = np.random.uniform(-1, 1, size=[batchsize, self.z_dims]).astype('float32')

		x_dummy = np.zeros(x_r.shape, dtype='float32')
		c_dummy = np.zeros(c.shape, dtype='float32')
		z_dummy = np.zeros(z_p.shape, dtype='float32')
		y_dummy = np.zeros((batchsize, 1), dtype='float32')
		f_dummy = np.zeros((batchsize, 3*1024), dtype='float32')

		# Train autoencoder
		enc_loss,_,_ = self.enc_trainer.train_on_batch([x_r, c, z_p], [x_dummy, z_dummy])
		print('a:', enc_loss)

		# Train generator
		g_loss, _, _, _, _, _, g_acc = self.dec_trainer.train_on_batch([x_r, c, z_p], [x_dummy, f_dummy, f_dummy])
		print('g:', g_loss, g_acc)

		# Train classifier
		cl_loss = self.cls_trainer.train_on_batch([x_r, c], c_dummy)
		print('c:', cl_loss)

		# Train discriminator
		d_loss, d_acc = self.dis_trainer.train_on_batch([x_r, c, z_p], y_dummy)
		print('d:', d_loss, d_acc)
		
		loss = {
			'g_loss': g_loss,
			'd_loss': d_loss,
			'g_acc': g_acc,
			'd_acc': d_acc
		}
		
		return loss
	
	def predict(self, z_samples):
		return self.f_dec.predict(z_samples)
	
	def build_model(self):
		print('Build models...')
		self.f_enc = self.build_encoder(output_dims=self.z_dims*2)
		self.f_dec = self.build_decoder()
		self.f_dis = self.build_discriminator()
		self.f_cls = self.build_classifier()
		
		#self.f_dis.summary()
		
		# Algorithm
		x_r = Input(shape=self.input_shape)
		c = Input(shape=(self.num_attrs,))
		z_params = self.f_enc([x_r, c])
		
		z_avg = Lambda(lambda x: x[:, :self.z_dims], output_shape=(self.z_dims,))(z_params)
		z_log_var = Lambda(lambda x: x[:, self.z_dims:], output_shape=(self.z_dims,))(z_params)
		z = Lambda(sample_normal, output_shape=(self.z_dims,))([z_avg, z_log_var])
		
		kl_loss = KLLossLayer()([z_avg, z_log_var])
		
		z_p = Input(shape=(self.z_dims,))
		
		x_f = self.f_dec([z, c])
		x_p = self.f_dec([z_p, c])
		
		y_r, f_D_x_r = self.f_dis(x_r)
		y_f, f_D_x_f = self.f_dis(x_f)
		y_p, f_D_x_p = self.f_dis(x_p)
		
		d_loss = DiscriminatorLossLayer()([y_r, y_f, y_p])
		
		c_r, f_C_x_r = self.f_cls(x_r)
		c_f, f_C_x_f = self.f_cls(x_f)
		c_p, f_C_x_p = self.f_cls(x_p)
		
		g_loss = GeneratorLossLayer()([x_r, x_f, f_D_x_r, f_D_x_f, f_C_x_r, f_C_x_f])
		gd_loss = FeatureMatchingLayer()([f_D_x_r, f_D_x_p])
		gc_loss = FeatureMatchingLayer()([f_C_x_r, f_C_x_p])
		
		c_loss = ClassifierLossLayer()([c, c_r])
		
		# Build classifier trainer
		set_trainable(self.f_enc, False)
		set_trainable(self.f_dec, False)
		set_trainable(self.f_dis, False)
		set_trainable(self.f_cls, True)
		
		print('Build classifier...')
		self.cls_trainer = Model(inputs=[x_r, c],
								 outputs=[c_loss])
		self.cls_trainer.compile(loss=[zero_loss],
								 optimizer=Adam(lr=1e-3))
		self.cls_trainer.summary()
		
		# Build discriminator trainer
		set_trainable(self.f_enc, False)
		set_trainable(self.f_dec, False)
		set_trainable(self.f_dis, True)
		set_trainable(self.f_cls, False)
		
		print('Build discriminator...')
		self.dis_trainer = Model(inputs=[x_r, c, z_p],
								 outputs=[d_loss])
		self.dis_trainer.compile(loss=[zero_loss],
								 optimizer=Adam(lr=1e-3),
								 metrics=[discriminator_accuracy(y_r, y_f, y_p)])
		self.dis_trainer.summary()
		
		# Build generator trainer
		set_trainable(self.f_enc, False)
		set_trainable(self.f_dec, True)
		set_trainable(self.f_dis, False)
		set_trainable(self.f_cls, False)
		
		print('Build generator...')
		self.dec_trainer = Model(inputs=[x_r, c, z_p],
								 outputs=[g_loss, gd_loss, gc_loss])
		self.dec_trainer.compile(loss=[zero_loss, zero_loss, zero_loss],
								 optimizer=Adam(lr=1e-3),
								 metrics=[generator_accuracy(y_p, y_f)])
		self.dec_trainer.summary()
		
		# Build autoencoder
		set_trainable(self.f_enc, True)
		set_trainable(self.f_dec, False)
		set_trainable(self.f_dis, False)
		set_trainable(self.f_cls, False)
		
		print('Build autoencoder...')
		self.enc_trainer = Model(inputs=[x_r, c, z_p],
								outputs=[g_loss, kl_loss])
		self.enc_trainer.compile(loss=[zero_loss, zero_loss],
								optimizer=Adam(lr=1e-3))
		self.enc_trainer.summary()
	
	def build_encoder(self, output_dims):
		input_img = Input(shape=self.input_shape)
		c_inputs = Input(shape=(self.num_attrs,))
		c = Reshape((1, 1, self.num_attrs))(c_inputs)
		#input_code = Input(shape=self.code_shape)
		flatten_code = Flatten()(c)
		
		x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
		x = add_units_to_conv2d(x, flatten_code)
		#print('K.int_shape(x): ', K.int_shape(x)) #  size here: (17, 32, 32)
		x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
		x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
		x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
		x = Flatten()(x)
		x = Dense(1024, activation='relu')(x)
		#z_mean = Dense(latent_dim)(x)
		#z_log_var = Dense(latent_dim)(x)
		
		x = Dense(output_dims)(x)
		e = Activation('linear')(x)
		
		return Model([input_img, c_inputs], e)
	
	def build_decoder(self):
		z_inputs = Input(shape=(self.z_dims, ))
		c_inputs = Input(shape=(self.num_attrs,))
		#input_code = Input(shape=self.code_shape)
		#flatten_code = Flatten()(input_code)
		x = concatenate([z_inputs, c_inputs])
		
		x = Dense(64*m_height*m_width)(x)
		#x = BatchNormalization()(x)
		x = Activation('relu')(x)
		#x = Dropout(0.3)(x)
		x = Reshape((64, m_height, m_width))(x)
		x = Conv2DTranspose(64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu')(x)
		x = Conv2DTranspose(32, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu')(x)
		x = Conv2D(channels, (5, 5), padding='same')(x)
		decoded = Activation('tanh')(x)
		
		return Model([z_inputs, c_inputs], decoded)
	
	def build_discriminator(self):
		input_img = Input(shape=self.input_shape)
		
		x = Conv2D(256, (7, 7), padding='same')(input_img)
		#x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		#x = Dropout(0.3)(x)
		x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
		
		x = Conv2D(128, (5, 5), padding='same')(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
		x = Conv2D(64, (3, 3), padding='same')(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
		f = Flatten()(x)
		x = Dense(1)(f)
		x = Activation('sigmoid')(x)
		
		return Model(input_img, [x, f])
	
	def build_classifier(self):
		input_img = Input(shape=self.input_shape)
		
		x = Conv2D(128, (7, 7), padding='same')(input_img)
		#x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		#x = Dropout(0.3)(x)
		x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
		
		x = Conv2D(256, (5, 5), padding='same')(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
		x = Conv2D(512, (3, 3), padding='same')(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
		f = Flatten()(x)
		x = Dense(2048)(f)
		x = Activation('relu')(x)
		x = Dense(self.num_attrs)(x)
		x = Activation('softmax')(x)
		
		return Model(input_img, [x, f])
	
	def save_weights(self, comment=''):
		self.f_enc.save_weights('temp/cvaegan_f_enc'+comment+'.h5')
		self.f_dec.save_weights('temp/cvaegan_f_dec'+comment+'.h5')
		self.f_dis.save_weights('temp/cvaegan_f_dis'+comment+'.h5')
		self.f_cls.save_weights('temp/cvaegan_f_cls'+comment+'.h5')
	
	def load_weights(self, comment=''):
		self.f_enc.load_weights('temp/cvaegan_f_enc'+comment+'.h5')
		self.f_dec.load_weights('temp/cvaegan_f_dec'+comment+'.h5')
		self.f_dis.load_weights('temp/cvaegan_f_dis'+comment+'.h5')
		self.f_cls.load_weights('temp/cvaegan_f_cls'+comment+'.h5')


cvaegan = CVAEGAN()

def get_image_from_net_data(data):
	res = data.transpose((1, 2, 0))
	#res *= 255.0
	res *= 127.5
	res += 127.5
	res = np.array(res, dtype=np.uint8)
	return res

def save_images(generated_images, dst='temp/cvaegan', comment=''):
	image = np.zeros((height, width*batch_size, channels), dtype=generated_images.dtype)
	for index, img in enumerate(generated_images):
		image[0:height, index*width:(index+1)*width] = get_image_from_net_data(img)
	cv2.imwrite(dst+'_'+comment+'.png', image)

# original images for test
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

orig_flabels = []
for l in labels:
	orig_flabels.append( l.flatten() )
orig_flabels = np.array(orig_flabels, dtype=np.float32)
print(orig_flabels.shape)


epochs = 1700

for epoch in range(epochs):
	print('Epoch {} from {} ...'.format(epoch, epochs))
	
	n = x_train.shape[0]
	indexes = np.random.randint(0, n, size=batch_size)
	image_batch = x_train[indexes,:,:,:]
	label_batch = y_train[indexes,:]
	
	if epoch % 10 == 0:
		print('Save gens ...')
		z_p = np.random.uniform(-1, 1, size=[batch_size, cvaegan.z_dims]).astype('float32')
		decoded_imgs = cvaegan.predict([z_p, orig_flabels])
		save_images(decoded_imgs, comment=str(epoch)+'_decoded')
		cvaegan.save_weights()
	
	if epoch % 100 == 0:
		cvaegan.save_weights(comment='_'+str(epoch))
	
	cvaegan.train_on_batch([image_batch, label_batch])


#cvaegan.load_weights()

'''
i = 0
for label in orig_flabels:
	i += 1
	lbls = []
	for j in range(batch_size):
		lbls.append(label)
	lbls = np.array(lbls, dtype=np.float32)
	print('lbls:', i, lbls.shape)
	
	#z_p = np.random.uniform(-1, 1, size=[batch_size, cvaegan.z_dims]).astype('float32')
	#z_imgs = cvaegan.predict([z_p, lbls])
	#save_images(z_imgs, dst='temp/cvaegan_z', comment=str(i))
		
	z = cvaegan.f_enc.predict([orig_images, lbls], batch_size=batch_size)
	z_p = z[:, :cvaegan.z_dims]
	stt_imgs = cvaegan.predict([z_p, lbls])
	save_images(stt_imgs, dst='temp/cvaegan_stt', comment=str(i))
'''

# experiments with new recipes
'''
ingred = np.zeros(code_h*code_w, dtype='float32')
lbls = []
for j in range(batch_size):
	lbls.append(ingred)
lbls = np.array(lbls, dtype=np.float32)
print('lbls:', i, lbls.shape)

z_p = np.random.uniform(-1, 1, size=[batch_size, cvaegan.z_dims]).astype('float32')
z_imgs = cvaegan.f_dec.predict([z_p, lbls], batch_size=batch_size)
save_images(z_imgs, dst='temp/cvaegan_new_recipes', comment='zeros')

for ing in range(code_w):
	ingcode = onehotencoder.transform(ing)
	print('ingred:', ingcode)
	ingcode = np.array([ingcode.flatten()]*code_h, dtype=np.float32).flatten()
	lbls = []
	for j in range(batch_size):
		lbls.append(ingcode)
	lbls = np.array(lbls, dtype=np.float32)
	print('lbls:', i, lbls.shape)

	z_imgs = cvaegan.f_dec.predict([z_p, lbls], batch_size=batch_size)
	save_images(z_imgs, dst='temp/cvaegan_new_recipes', comment=str(ing))

no = np.zeros(code_w, dtype='float32')
ing1 = onehotencoder.transform(2).flatten()
ing2 = onehotencoder.transform(7).flatten()
ing3 = onehotencoder.transform(10).flatten()
ing4 = onehotencoder.transform(20).flatten()

ingcode = np.array([ing1, ing2, ing3, ing4, no, no, no, no, no], dtype=np.float32).flatten()
lbls = []
for j in range(batch_size):
	lbls.append(ingcode)
lbls = np.array(lbls, dtype=np.float32)
print('lbls:', i, lbls.shape)

z_imgs = cvaegan.f_dec.predict([z_p, lbls], batch_size=batch_size)
save_images(z_imgs, dst='temp/cvaegan_new_recipes', comment='-2-7-10-20-no')
'''

if __name__ == '__main__':
	print('Start...')
	print('Theano version: {}'.format(theano_version))
	print('Keras version: {}'.format(keras_version))
