#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
get data from Dodopizza site
'''

__author__ = 'noonv'

import requests
from requests import Session
import bs4

import os
import sys

import pandas as pd

base_url = 'https://dodopizza.ru'

headers = {
	'user-agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'
}


def get_cities(siteurl):
	print('Open site', siteurl)
	
	res = requests.get(siteurl, headers = headers)
	res.raise_for_status()
	
	soup = bs4.BeautifulSoup(res.text, 'lxml')
	
	print('Get cities...')
	cities_list = soup.find('div', {'class': 'cities_list cities_list_active'})
	cities = cities_list.find_all('div', {'class': 'cities_list_item'})
	print('Cities count', len(cities))
	
	citi_names = []
	citi_urls = []
	for citi in cities:
		citi_name = citi.find('a').getText()
		citi_url = citi.find('a').get('href')
		print(citi_name, citi_url)
		citi_names.append(citi_name)
		citi_urls.append(citi_url)
	
	return citi_names, citi_urls

def get_pizzas_list(siteurl):
	print('Open site', siteurl)
	
	res = requests.get(siteurl, headers = headers)
	res.raise_for_status()
	print(res.status_code)
	
	soup = bs4.BeautifulSoup(res.text, 'lxml')
	
	prod_pizzas = soup.find('div', {'class': 'prod_pizzas'})
	
	items = prod_pizzas.find_all('div', {'class': 'prod_block'})
	print('Pizzas count', len(items))
	
	pizza_names = []
	pizza_eng_names = []
	pizza_urls = []
	
	for item in items:
		pizza_link = item.find('h3', {'class': 'prod_header'}).find('a').get('href')
		pizza_link = base_url + pizza_link
		print('link:', pizza_link)
		pizza_eng_name = pizza_link.split('/')[-1]
		print('eng name:', pizza_eng_name)
		pizza_name = item.find('h3', {'class': 'prod_header'}).find('a').find('span').getText()
		print('name:', pizza_name)
		pizza_contain = item.find('div', {'class': 'prod_contains'}).getText().strip()
		print('contain:', pizza_contain)
		#get_pizza_data(pizza_link)
		
		pizza_names.append(pizza_name)
		pizza_eng_names.append(pizza_eng_name)
		pizza_urls.append(pizza_link)
	
	return pizza_names, pizza_eng_names, pizza_urls


def get_pizza_data(siteurl):
	print('Open pizza data url', siteurl)
	
	#res = requests.get(siteurl, headers = headers)
	session = Session()
	session.head(siteurl)
	
	headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36',
		'Referer': siteurl,
		'x-requested-with': 'XMLHttpRequest'
    }
	res = session.get(siteurl, headers = headers)
	
	res.raise_for_status()
	print(res.status_code)
	#print(res.headers)
	#print(len(res.content))
	#print(res.content)
	
	soup = bs4.BeautifulSoup(res.text, 'lxml')	
	
	pizza = soup.find('div', {'class': 'popup_prod_cont'})
	#print(pizza.getText())
	images = pizza.find('div', {'class': 'product_slides'}).find('ul', {'class': 'slides'}).find_all('li')
	print('Images count:', len(images))
	
	image_urls = []
	for item in images:
		image_src = item.find('img').get('src')
		print('image:', image_src)
		image_urls.append(image_src)
	
	pizza_contain = pizza.find('div', {'class': 'prod_contains'}).getText().strip()
	print('contain:', pizza_contain)
	
	kiloCalories = pizza.find('span', {'id': 'kiloCalories'}).getText()
	carbohydrates = pizza.find('span', {'id': 'carbohydrates'}).getText()
	proteins = pizza.find('span', {'id': 'proteins'}).getText()
	fats = pizza.find('span', {'id': 'fats'}).getText()
	print('valuetable:', kiloCalories, carbohydrates, proteins, fats)
	
	size = pizza.find('span', {'id': 'sizeValue'}).getText()
	weight = pizza.find('span', {'id': 'weightValue'}).getText()
	print('size&:weight', size, weight)
	
	pizza_params = {'kiloCalories': kiloCalories, 'carbohydrates': carbohydrates, 'proteins': proteins, 'fats': fats, 'size': size, 'weight': weight}
	
	pizza_price = pizza.find('div', {'class': 'prod_price_value'}).find('span', {'class': 'value'}).getText()
	print('price:', pizza_price)
	
	return image_urls, pizza_contain, pizza_price, pizza_params

saved_urls = []

def download_files(urls, dir_to_save, prefix=''):
	print('Download files...')
	
	global saved_urls
	
	counter = 0
	for url in urls:
		counter += 1
		
		if url in saved_urls:
			print('Allready downloaded! {}'.format(url))
			continue
		
		print('Downloading {} ...'.format(url))
		res = requests.get(url, headers = headers)
		res.raise_for_status()
		
		os.makedirs(dir_to_save, exist_ok=True)
		
		filename = os.path.basename(url)
		extension = os.path.splitext(filename)[1]
		filename = prefix + str(counter) + extension
		print('Save file to {} ...'.format(filename))
		
		file = open(os.path.join(dir_to_save, filename), 'wb')
		for chunk in res.iter_content(100000):
			file.write(chunk)
		file.close()
		
		saved_urls.append(url)

def grab_dodopizza_site():
	print('Grab dodopizza site...')
	
	city_names, city_urls = get_cities(base_url)
	
	# save cities info
	print('Save cities info...')
	df_city = pd.DataFrame()
	df_city['city'] = city_names
	df_city['city_url'] = city_urls
	print( df_city.shape )
	df_city.to_csv('cities.csv', index=False)

	# Just one city pizzas
	city_names, city_urls = ['Калининград'], ['/Kaliningrad']
	
	pizzas = []
	
	for city_counter in range(0, len(city_names)):
		city_name = city_names[city_counter]
		city_url = city_urls[city_counter]
		
		# get pizzas list
		pizza_names, pizza_eng_names, pizza_urls = get_pizzas_list(base_url+city_url)
		
		# save city pizzas info
		print('Save city pizzas info...')
		df_city_pizzas = pd.DataFrame()
		df_city_pizzas['pizza_name'] = pizza_names
		df_city_pizzas['pizza_name_eng'] = pizza_eng_names
		df_city_pizzas['pizza_url'] = pizza_urls
		print( df_city_pizzas.shape )
		
		city_url2 = city_url.replace('/', '')
		df_city_pizzas.to_csv(city_url2+'-pizzas.csv', index=False)
		
		# get pizza information and photos
		for i in range(0, len(pizza_names)):
			pizza_name = pizza_names[i]
			pizza_eng_name = pizza_eng_names[i]
			pizza_url = pizza_urls[i]
			
			# ignore Combo
			if pizza_url.find("ComboDetails") != -1:
				print('Skip Combo!')
				continue
			
			image_urls, pizza_contain, pizza_price, pizza_params = get_pizza_data(pizza_url)
			download_files(image_urls, pizza_eng_name, pizza_eng_name)
			
			print('pizza_params:', pizza_params)
			
			# город, URL города, название, названиеENG, URL пиццы, содержимое, цена, калории, углеводы, белки, жиры, диаметр, вес
			pizza = [ city_name, city_url, pizza_name, pizza_eng_name, pizza_url, pizza_contain, pizza_price, 
				pizza_params['kiloCalories'], pizza_params['carbohydrates'], pizza_params['proteins'], 
				pizza_params['fats'], pizza_params['size'], pizza_params['weight'] ]
			pizzas.append(pizza)
	
	# save pizzas info
	print('Save pizzas info...')
	pizzas_columns = ['city_name', 'city_url', 'pizza_name', 'pizza_eng_name', 'pizza_url', 'pizza_contain', 'pizza_price', 
		'kiloCalories', 'carbohydrates', 'proteins', 'fats', 'size', 'weight']
	
	df_pizza = pd.DataFrame(pizzas, columns=pizzas_columns)	
	print( df_pizza.head() )
	print( df_pizza.shape )
	df_pizza.to_csv('pizzas.csv', index=False)
	
if __name__ == '__main__':
	print('Start...')
	
	grab_dodopizza_site()

	print('Done.')
