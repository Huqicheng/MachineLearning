import os
import numpy as np

#convert string to float
def strToFloat(float_string):
    return float(float_string)

#read feature matrix of training samples(n-dim)
def read_x(file_path):
	f = open(file_path,'r')
	data_x = []
	for line in f:
		list_data_str =line.split(',')
		list_data = []
		for data in list_data_str:
			list_data.append(strToFloat(data.strip()))
		data_x.append(list_data)
	return data_x

#read result of training samples (1-dim)
def read_y(file_path):
	f = open(file_path,'r')
	data_y = []
	for line in f:
		data_y.append(strToFloat(line.strip()))
	return data_y



