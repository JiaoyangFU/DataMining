import numpy as np
import math
import csv

def calRMSE(prediction):	#prediction:{(uid, bid):(predict_value, true_value)}
	rmse = 0.
	count = 0
	for pair in prediction:
		(uid,bid) = pair
		(predict, ture) = prediction[pair]
		rmse += (predict - ture) ** 2
		count += 1
	rmse = np.sqrt(rmse/count)
	return rmse

def calMAE(prediction):
	mae = 0.
	count = 0
	for pair in prediction:
		(predict,true) = prediction[pair]
		mae += abs(predict - true)
		count += 1
	mae = np.sqrt(mae/count)
	return mae

