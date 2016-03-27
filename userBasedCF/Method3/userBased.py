# using pearson Similarity method

import pandas as pd
import numpy as np
import math
import random



def getMean(user_dict,user):
    total=0.0
    cnt=0
    
    for item in user_dict[user]:
        total += user_dict[user][item]
        cnt += 1
    
    return total/cnt


def make_prediction(user_dict, restaurant, user,k,similarityThreshold):
    similar_dict = topMatches(user_dict, user, k, similarity=pearsonSimilarity)
    numerator = 0.0
    denominator = 0.0
    userMean=getMean(user_dict,user)
    
    denominator = sum([ similar_dict[user] for user in similar_dict])
    
    if abs(denominator) <=0.000000001:
        return userMean

    for otherUser in similar_dict:
        if restaurant in user_dict[otherUser]:
            similarity = similar_dict[otherUser]
            otherUserMean = getMean(user_dict,otherUser)
            numerator +=similarity * (user_dict[otherUser][restaurant]-otherUserMean)
    
    return userMean+ numerator/denominator
    



def pearsonSimilarity(user_i, user_j, user_dict):
    commonItemList=[]
    
    for item in user_dict[user_i]:
        if item in  user_dict[user_j]:
            commonItemList.append(item)
    N = len(commonItemList)
    if N == 0:
        return 0
    
    sum1 = sum([float(user_dict[user_i][item]) for item in commonItemList])
    sum2 = sum([float(user_dict[user_j][item]) for item in commonItemList])
    square_sum1 = sum([pow(user_dict[user_i][item],2) for item in commonItemList])
    square_sum2 = sum([pow(user_dict[user_j][item],2) for item in commonItemList])

    product_sum = sum([user_dict[user_i][item]* user_dict[user_i][item] for item in commonItemList])

    numerator = product_sum- (sum1 * sum2/N)
    denominator = math.sqrt((square_sum1- pow(sum1,2)/N )*(square_sum2- pow(sum2,2)/N))

    if denominator==0:
        return 0

    return numerator/denominator



def topMatches(user_dict,user_to_predict,k,similarity=pearsonSimilarity):
    scores=[(similarity(user_to_predict,other_user,user_dict), other_user) for other_user in user_dict  if other_user != user_to_predict]
    scores.sort()    
    scores.reverse()    
    scores = scores[0:k]
    result = {}
    for entry in scores:
        result[entry[1]] = entry[0] #{user,similarity}
    return result


'''
    k is the number of neighbors
    similarityThreshold is the minimum similarity can be considered
'''
def CF_evaluating(test_user_data, user_dict,k,similarityThreshold):
    evaluations = dict()
    for user in test_user_data:
        for restaurant, rating in test_user_data[user].items():
            true = rating
            pair = (user, restaurant)
            prediction = make_prediction(user_dict, restaurant, user,k,similarityThreshold)
            evaluations[pair] = (prediction, true)
    return evaluations


#test_user_data  => {user_id : {business_id : [rating]}}

#user_dict
#=> {user_id : {business_id : [rating]}} that can be indexed by user_id







