#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
get pizza data
'''

__author__ = 'noonv'

import os
import sys
import numpy as np
np.random.seed(42)

import pandas as pd
import cv2

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def read_data():
	print('Read csv...')
	df = pd.read_csv('pizzas.csv', encoding='cp1251')
	print(df.shape)
	print(df.info())
	
	return df


def get_pizza_names(df):
	# Get names
	pizza_names = df['pizza_name'].tolist()
	pizza_eng_names = df['pizza_eng_name'].tolist()
	return pizza_names, pizza_eng_names

def load_data():
	df = read_data()
	
	df = prepare_data(df)
	print(df.head())
	print(df.describe())
	
	# Get names
	pizza_names, pizza_eng_names = get_pizza_names(df)
	print( pizza_eng_names )
	
	ingredients, ingredients_count = normalize_contains(df)
	
	min_count = np.min(ingredients_count)
	print('min:', min_count)
	max_count = np.max(ingredients_count)
	print('max:', max_count)
	
	df_ingredients = pd.DataFrame(ingredients)
	df_ingredients.fillna(value='0', inplace=True)
	print(df_ingredients)
	print(df_ingredients.describe())
	
	print(df_ingredients.stack().value_counts())
	
	# Make encoding...
	ingredients_full = df_ingredients.values.tolist()

	# flatten lists
	flat_ingredients = [item for sublist in ingredients_full for item in sublist]
	print(flat_ingredients)
	print(len(flat_ingredients))

	np_ingredients = np.array(flat_ingredients)
	#print(np_ingredients)

	labelencoder = LabelEncoder()
	ingredients_encoded = labelencoder.fit_transform(np_ingredients)
	print(ingredients_encoded)

	label_max = np.max(ingredients_encoded)
	print('max:', label_max)
	
	for label in range(label_max):
		print(label, labelencoder.inverse_transform(label))
	
	lb_ingredients = []
	for lst in ingredients_full:
		lb_ingredients.append(labelencoder.transform(lst).tolist())
	#lb_ingredients = np.array(lb_ingredients)
	print( lb_ingredients )
	
	onehotencoder = OneHotEncoder(sparse=False)
	ingredients_onehotencoded = onehotencoder.fit_transform(ingredients_encoded.reshape(-1, 1))
	print(ingredients_onehotencoded.shape)
	
	df_ingredients_encoded = pd.DataFrame(lb_ingredients)
	df_ingredients_encoded
	
	#print(df_ingredients_encoded.describe())
	
	labels = []
	for label in lb_ingredients:
		lb = np.array(label)
		lbo = onehotencoder.transform( lb.reshape(-1, 1) )
		labels.append(lbo)
	
	return labels, onehotencoder


def prepare_data(df):
	df['kiloCalories'] = df.kiloCalories.apply(lambda x: x.replace(',','.'))
	df['carbohydrates'] = df.carbohydrates.apply(lambda x: x.replace(',','.'))
	df['proteins'] = df.proteins.apply(lambda x: x.replace(',','.'))
	df['fats'] = df.fats.apply(lambda x: x.replace(',','.'))
	df['weight'], df['weight_err'] = df['weight'].str.split('±', 1).str
	
	df['kiloCalories'] = df.kiloCalories.astype('float32')
	df['carbohydrates'] = df.carbohydrates.astype('float32')
	df['proteins'] = df.proteins.astype('float32')
	df['fats'] = df.fats.astype('float32')
	df['weight'] = df.weight.astype('int64')
	df['weight_err'] = df.weight_err.astype('int64')
	
	df['pizza_kiloCalories'] = df.kiloCalories * df.weight / 100
	df['pizza_carbohydrates'] = df.carbohydrates * df.weight / 100
	df['pizza_proteins'] = df.proteins * df.weight / 100
	df['pizza_fats'] = df.fats * df.weight / 100
	
	return df
	
def normalize_contains(df):
	print('Normalize contains...')
	
	'''
	def split_contain(contain):
		lst = contain.split(',')
		print(len(lst),':', lst)

	for i, row in df.iterrows():
		split_contain(row.pizza_contain)
	'''
	
	def split_contain2(contain):
		lst = contain.split(',')
		#print(len(lst),':', lst)
		for i in range(len(lst)):
			item = lst[i]
			item = item.replace('увеличенная порция', '')
			item = item.replace('увеличенные порции', '')
			item = item.replace('сыра моцарелла', 'моцарелла')
			item = item.replace('моцареллы', 'моцарелла')
			item = item.replace('цыпленка', 'цыпленок')
			and_pl = item.find(' и ')
			if and_pl != -1:
				item1 = item[0:and_pl]
				item2 = item[and_pl+3:]
				item = item1
				lst.insert(i+1, item2.strip())
			double_pl = item.find('двойная порция ')
			if double_pl != -1:
				item = item[double_pl+15:]
				lst.insert(i+1, item.strip())
			lst[i] = item.strip()
		# last one
		for i in range(len(lst)):
			lst[i] = lst[i].strip()
		print(len(lst),':', lst)
		return lst

	ingredients = []
	ingredients_count = []
	for i, row in df.iterrows():
		print(row.pizza_name)
		lst = split_contain2(row.pizza_contain)
		ingredients.append(lst)
		ingredients_count.append(len(lst))
	print(ingredients_count)
	
	return ingredients, ingredients_count


def load_images(image_paths):
	print('Load images...')
	# load images
	images = []
	for path in image_paths:
		print('Load image:', path)
		image = cv2.imread(path)
		if image is not None:
			images.append(image)
		else:
			print('Error read image:', path)
	
	return images

def cut_pizza_from_images(images):
	print('Cut pizza from images...')
	pizza_imgs = []
	for img in images:
		y, x, height, width = 0, 165, 380, 380
		pizza_crop = img[y:y+height, x:x+width]
		pizza_imgs.append(pizza_crop)
	print(pizza_imgs[0].shape)
	print(len(pizza_imgs))
	
	return pizza_imgs


def rotate(image, angle):
	rows1, cols1, channels1 = image.shape
	
	rb, cb = int(rows1/4), int(cols1/4)
	img = cv2.copyMakeBorder(image, rb, rb, cb, cb, cv2.BORDER_REPLICATE) # top, bottom, left, right 
	
	rows, cols, channels = img.shape
	
	M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
	dst = cv2.warpAffine(img, M, (cols, rows))	
	res = dst[rb:rb+rows1, cb:cb+cols1]
	
	return res


def resize_rotate_flip(image, new_size, rotate_angles=range(0, 360), make_flip=True, flip_param=1):
	image = cv2.resize(image, new_size)
	
	image_list = []
	for angle in rotate_angles:
		img = rotate(image, angle)
		image_list.append(img)
		if make_flip == True:
			img2 = cv2.flip(img, flip_param) # 0 - horizontal flip, 1 - vertical flip
			image_list.append(img2)
	
	return image_list


if __name__ == '__main__':
	print('Start...')
	
	load_data()
	print('Done.')

