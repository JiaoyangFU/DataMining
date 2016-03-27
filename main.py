import sys
import loadData
import baseline
import evaluation
import Matrix_Factorization
import itemBased
import csv

def build_dict(dataFrame, user_restaurant_dict, restaurant_user_dict):
    """
    user_restaurant_dict is a new review dictionary {user_id : {business_id : [review]}} that can be indexed by user_id
    restaurant_user_dict is a new review dictionary {user_id : {business_id : [review]}} that can be indexed by restaurant_id
    """
    assert len(user_restaurant_dict) == 0, "user_restaurant_dict must be empty"
    assert len(restaurant_user_dict) == 0, "restaurant_user_dict must be empty"
    for _, row in dataFrame.iterrows():
        u_temp = row.user_id
        i_temp = row.business_id
        rating = float(row.stars)
        if u_temp not in user_restaurant_dict:
            user_restaurant_dict[u_temp] = dict()
        user_restaurant_dict[u_temp][i_temp] = rating
        if i_temp not in restaurant_user_dict:
            restaurant_user_dict[i_temp] = dict()
        restaurant_user_dict[i_temp][u_temp] = rating

def build_test_set(user_restaurant_dict, required_review_num):
    """
    give user_restaurant_dict dictionary and required_review_num, 
    return a set of user_id that only contains users who has more than required_review_num reviews
    """
    tested_uid = set()
    for user, reviews in user_restaurant_dict.items():
        if len(reviews) >= required_review_num:
            tested_uid.add(user)
    return tested_uid


def build_train_test_data(user_restaurant_dict, restaurant_user_dict, test_set, test_percentage):
    """
    update user/restaurant_user_dict based on test_set, set aside test_percentage reviews from users in test_set for testing purposes
    return the dictionary that contains testing data
    """
    test_data = dict()
    for test_user in test_set:
        test_data[test_user] = dict()
        total_review_num = len(user_restaurant_dict[test_user])
        test_review_num = int(total_review_num * test_percentage)
        for i, (restaurant, rating) in enumerate(user_restaurant_dict[test_user].items()):
            if len(restaurant_user_dict[restaurant]) == 1: # restaurant only has one review, don't delete it
                test_review_num += 1 # go to next item
                continue
            test_data[test_user][restaurant] = rating
            del user_restaurant_dict[test_user][restaurant]
            del restaurant_user_dict[restaurant][test_user]
            if i == test_review_num:
                break 
    return test_data

def cal_average_rating(user_restaurant_dict):
    """
    for a given user, calculate this user's average rating for all the reviews in user_restaurant_dict table
    """
    total = 0.
    count = 0
    for user in user_restaurant_dict:
    	for restaurant, rating in user_restaurant_dict[user].items():
    		total += rating
    		count += 1
    return total/count


def main(argv):

    review_minimum_num = 30
    test_percentage = 0.2 

    # initialize variable    
    """Store data into two dictionary"""
    user_restaurant_dict = {} # {user_id: {business_id: rating}
    restaurant_user_dict = {}  # {business_id: {user_id :rating}}
    smalldf = loadData.Parser() #load data from json file
    # build reviews that can be indexed from both user_id and restaurant_id 
    print "Store data in dictionaries..."
    build_dict(smalldf, user_restaurant_dict, restaurant_user_dict)   
    #extract users with more than review_minimum_num reviews
    test_set = build_test_set(user_restaurant_dict, review_minimum_num)
    #extract test data from orignal dataset
    test_data = build_train_test_data(user_restaurant_dict, restaurant_user_dict, test_set, test_percentage)
    print "total number of users in test_data:", len(test_data)
    number_of_users = len(user_restaurant_dict)
    number_of_restaurants = len(restaurant_user_dict)
    print "total number of users in training data:", len(user_restaurant_dict)
    print "total number of restaurants in training data:", len(restaurant_user_dict)

    """user_restaurant_dict: training dataset
       test_data: test dataset
    """
    #baseline prediction and evaluation
    base_evaluation = baseline.base_evaluating(test_data, user_restaurant_dict, restaurant_user_dict)
    base_rmse = evaluation.calRMSE(base_evaluation)
    print "baseline rmse for test data is:", base_rmse

    CF_evaluations = itemBased.CF_evaluating(test_data, restaurant_user_dict)
    CF_rmse = evaluation.calRMSE(CF_evaluations)
    print "Item-based CF rmse for test data is:",CF_rmse

    svd_method = Matrix_Factorization.svd(number_of_users, number_of_restaurants, user_restaurant_dict, 20, test_data)
    svd_rmse = evaluation.calRMSE(svd_method)
    print "SVD rmse for test data is:", svd_rmse

if __name__ == '__main__':
    main(sys.argv)
