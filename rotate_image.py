#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
rotate pizza photo
'''

__author__ = 'noonv'

import os
import sys
import cv2

def rotate(image, angle):
	rows1, cols1, channels1 = image.shape
	
	rb, cb = int(rows1/4), int(cols1/4)
	img = cv2.copyMakeBorder(image, rb, rb, cb, cb, cv2.BORDER_REPLICATE) # top, bottom, left, right 
	
	rows, cols, channels = img.shape
	
	M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
	dst = cv2.warpAffine(img, M, (cols, rows))	
	res = dst[rb:rb+rows1, cb:cb+cols1]
	
	return res


def rotate_image(path, dir_to_save='./images/rotate'):
	print('Rotate image...')
	
	print('Load image:', path)
	image = cv2.imread(path)
	assert(image is not None)
	
	flbase = os.path.basename(path)
	filename, file_extension = os.path.splitext(flbase)
	
	os.makedirs(dir_to_save, exist_ok=True)
	
	#image = cv2.resize(image, (224, 224)) 
	
	# add image border (to avoid black borders by rotate image)
	rows1, cols1, channels1 = image.shape
	
	rb, cb = int(rows1/4), int(cols1/4)
	img = cv2.copyMakeBorder(image, rb, rb, cb, cb, cv2.BORDER_REPLICATE) # top, bottom, left, right 
	
	rows, cols, channels = img.shape
	
	for angle in range(1, 180): #360):
		M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
		dst = cv2.warpAffine(img, M, (cols, rows))
		
		res = dst[rb:rb+rows1, cb:cb+cols1]
		
		filename2 = filename + str(angle) + file_extension
		path2 = os.path.join(dir_to_save, filename2)
		print('Save image {} ...'.format(path2))
		cv2.imwrite(path2, res)


if __name__ == '__main__':
	print('Start...')
	
	rotate_image('./images/chizburger-pizza.png')

	print('Done.')

