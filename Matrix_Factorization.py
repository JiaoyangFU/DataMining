import random, math
import numpy as np


def svd(u_num, r_num, user_indexed_reviews, factor_num, test_user_data):
    """
    :param u_num: the number of the users
    :param r_num: the number of the restaurants
    :param factor_num: the number of features
    function: Initializing the vector p and q represents user and restaurants
    and training the p,q,b
    """

    u_num = u_num
    r_num = r_num
    factor_num = factor_num
    learning_rate = 0.001
    regularization = 0.05
    evaluations = dict()

    # calculating the average rating of the whole matrix
    count = 0
    sum_rating = 0.0
    for user in user_indexed_reviews:
        for restaurant, rating in user_indexed_reviews[user].items():
            sum_rating += rating
            count += 1
    avg_rating = sum_rating / count
    bu = [0.0 for i in range(u_num)]
    br = [0.0 for i in range(r_num)]

    # feature matrix of the user
    p = [[3.1 * random.random() / math.sqrt(factor_num) for i in range(factor_num)] \
         for j in range(u_num)]
    q = [[3.1 * random.random() / math.sqrt(factor_num) for i in range(factor_num)] \
         for j in range(r_num)]

    # finishing initialization

    # start training:
    for user_index, (user, restaurant_ratings) in enumerate(user_indexed_reviews.items()):
        for restaurant_index, (restaurant, rating) in enumerate(restaurant_ratings.items()):
            current_score = predictScore(avg_rating,bu[user_index],br[restaurant_index],p[user_index],q[restaurant_index])
            error = rating - current_score

            # update
            for k in range(factor_num):
                p_uk = p[user_index][k]
                p[user_index][k] += learning_rate * (error * q[restaurant_index][k] - regularization * p[user_index][k])
                q[restaurant_index][k] += learning_rate * (error * p_uk -  regularization * q[restaurant_index][k])
            bu[user_index] += learning_rate * (error - regularization * bu[user_index])
            br[restaurant_index] += learning_rate * (error - regularization * br[restaurant_index])
    # end of training

    # start testing
    for user_index, user in enumerate(test_user_data):
        for restaurant_index,(restaurant, rating) in enumerate(test_user_data[user].items()):
            true = rating
            pair = (user, restaurant)
            prediction = predictScore(avg_rating,bu[user_index],br[restaurant_index], p[user_index],q[restaurant_index])
            evaluations[pair] = (true, prediction)

    return evaluations

def innerProduct(v1, v2):
    """
    :param v1: Vector 1
    :param v2: Vector 2
    :return: the inner product of two vectors
    """
    innerP = 0.0
    for i in range(len(v1)):
        innerP += v1[i] * v2[i]
    return innerP


def predictScore(average, bu, br, pu, pr):
    """
    :param average:  whole average rating
    :param pu: vector user_u
    :param pr: vector restaurant_r
    :return: predictScore
    """
    score = average + bu + br + innerProduct(pu, pr)
    if score < 0.5:
        score = 0.5
    elif score > 5:
        score = 5
    return score
