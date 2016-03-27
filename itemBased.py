import pandas as pd
import numpy as np
import math
import random

def CF_evaluating(test_user_data, user_rating_table, item_table):
    """
    calculate evaluations using collaborative filtering
    test_user_data -- {user : {restaurant : [reviews]}}
    user_rating_table --{user : {restaurant : rating}}
    item_table -- {item : {user : rating - average}}
    return evaluations -- {user : {restaurant : (true_rating, prediction)}}
    """
    evaluations = dict() 
    count = 1
    for user in test_user_data:
        count += 1
        count_item = 1
        #evaluations[user] = dict()
        for restaurant, rating in test_user_data[user].items():
            count_item += 1
            true_rating = rating
            pair = (user, restaurant)
            #prediction = prediction(user_rating_table[user], item_table, restaurant, user)
            prediction = make_prediction(item_table, restaurant, user)
            evaluations[pair] = (prediction, true_rating)
    return evaluations

def make_prediction(item_table, restaurant, user):

    similar_rest = topMatches(item_table, restaurant, n = 150, similarity=pearson_sim)
    numerator, denominator = 0.0, 0.0
    for other in similar_rest:
        if user in item_table[other]:
            similarity = similar_rest[other]
            numerator += similarity * item_table[other][user]
            denominator += similarity
    if denominator == 0: 
        prediction = random.randint(1, 10) * 0.5#all_users[(user,)][0]["average_stars"]
    else:
        prediction = numerator/denominator
        if prediction < 0.5:
            prediction = 0.5
    return round(prediction*2)/2


def prediction(item_rating_table, item_table, item_to_predict, user):
    """
    item_rating_table {item : rating}} 
    item_table -- {item : {user : rating }} 
    with item in table all being items rated by the user, 
    return a predicated rating for item of the user
    """
    numerator, denominator = 0.0, 0.0
    for item in item_rating_table:
        similarity = pearson_sim(item_to_predict, item, item_table)
        numerator += similarity * item_rating_table[item]
        denominator += similarity
    if denominator == 0: 
        prediction = random.randint(1, 10) * 0.5
    else:
        prediction = numerator/denominator
        if prediction < 0.5:
            prediction = 0.5
    return round(prediction*2)/2

def pearson_sim(item_i, item_j, item_table):
    """
    give a dictionary item_table {item : {user : rating}}, 
    return the CF similarity of item_i, item_j
    """
    product, sum_square1, sum_square2 = 0.0, 0.0, 0.0
    for user in item_table[item_i]:
        if user in item_table[item_j]:
            product += item_table[item_i][user] * item_table[item_j][user]
            sum_square1 += item_table[item_i][user]**2
            sum_square2 += item_table[item_j][user]**2
    if sum_square1 == 0 or sum_square2 == 0:
        return 0
    else:
        return abs(product)/math.sqrt(sum_square1 * sum_square2)

def topMatches(item_table,item_to_predict,n=5,similarity=pearson_sim):    
    scores=[(similarity(item_to_predict,other,item_table),other) for other in item_table  if other != item_to_predict]    
    scores.sort()    
    scores.reverse()    
    scores = scores[0:n]
    res = {}
    for entry in scores:
        res[entry[1]] = entry[0]
    return res


