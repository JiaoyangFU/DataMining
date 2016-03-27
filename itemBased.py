import pandas as pd
import numpy as np
import math
import random


def make_prediction(restaurant_dict, restaurant, user):
    similar_rest = topMatches(restaurant_dict, restaurant, n = 10, similarity=sim_pearson)
    numerator = 0.
    denominator = 0.
    for other in similar_rest:
        if user in restaurant_dict[other]:
            similarity = similar_rest[other]
            numerator += similarity * restaurant_dict[other][user]
            denominator += similarity
    if denominator == 0: 
        prediction = random.randint(1, 10) * 0.5#all_users[(user,)][0]["average_stars"]
    else:
        prediction = numerator/denominator
        if prediction < 0.5:
            prediction = 0.5
    return round(prediction*2)/2

def sim_pearson(item_i, item_j, restaurant_dict):
    si = {}
    for user in restaurant_dict[item_i]:
        if user in restaurant_dict[item_j]:
            si[user] = 1
    n = len(si)
    if n == 0:
        return 0
    sum1 = sum([restaurant_dict[item_i][it] for it in si])
    sum2 = sum([restaurant_dict[item_j][it] for it in si])
    sum1Sq=sum([pow(restaurant_dict[item_i][it],2) for it in si])
    sum2Sq=sum([pow(restaurant_dict[item_j][it],2) for it in si])

    pSum=sum([restaurant_dict[item_i][it] * restaurant_dict[item_j][it] for it in si])
    numerator=pSum-(sum1*sum2/n)

    denominator=math.sqrt((sum1Sq-pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))
    if denominator==0:
        return 0
    r = numerator/denominator
    return r


def topMatches(restaurant_dict,item_to_predict,n=5,similarity=pearson_sim):    
    scores=[(similarity(item_to_predict,other,restaurant_dict),other) for other in restaurant_dict  if other != item_to_predict]    
    scores.sort()    
    scores.reverse()    
    scores = scores[0:n]
    res = {}
    for entry in scores:
        res[entry[1]] = entry[0]
    return res

def CF_evaluating(test_user_data, restaurant_dict):
    evaluations = dict() 
    count = 1
    for user in test_user_data:
        count += 1
        count_item = 1
        for restaurant, rating in test_user_data[user].items():
            count_item += 1
            true = rating
            pair = (user, restaurant)
            prediction = make_prediction(restaurant_dict, restaurant, user)
            evaluations[pair] = (prediction, true)
    return evaluations
