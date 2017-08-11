#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
cut pizza photos
'''

__author__ = 'noonv'

import os
import cv2

import load_data

def load_photos():
	df = load_data.read_data()
	
	# Get names
	pizza_names, pizza_eng_names = load_data.get_pizza_names(df)
	print( pizza_eng_names )
	
	# prepare image paths
	image_paths = []
	for name in pizza_eng_names:
		path = os.path.join(name, name+'3.jpg')
		image_paths.append(path)
	print(image_paths)
	
	images = load_data.load_images(image_paths)

	# cut pizza from photo
	pizza_imgs = load_data.cut_pizza_from_images(images)
	
	return pizza_eng_names, pizza_imgs


def save_photos(pizza_eng_names, pizza_imgs, dir_to_save = 'images'):
	print('Save images...')
	
	assert(len(pizza_eng_names) == len(pizza_imgs))
	
	os.makedirs(dir_to_save, exist_ok=True)
	
	for i in range( len(pizza_eng_names) ):
		name = pizza_eng_names[i]
		img = pizza_imgs[i]
		filename = name + '.png'
		path = os.path.join(dir_to_save, filename)
		print('Save image {} ...'.format(path))
		cv2.imwrite(path, img)
	


if __name__ == '__main__':
	print('Start...')
	
	pizza_eng_names, pizza_imgs = load_photos()
	
	save_photos(pizza_eng_names, pizza_imgs)

	print('Done.')

