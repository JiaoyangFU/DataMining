import sys
import loadData
import baseline
import evaluation

def build_user_and_restaurant_indexed_reviews(dataFrame, user_indexed_reviews, restaurant_indexed_reviews):
    """
    all_reviews is a dictionary {(user_id, business) : [review]}
    user_indexed_reviews is a new review dictionary {user_id : {business_id : [review]}} that can be indexed by user_id
    restaurant_indexed_reviews is a new review dictionary {user_id : {business_id : [review]}} that can be indexed by restaurant_id
    """
    assert len(user_indexed_reviews) == 0, "user_indexed_reviews must be empty"
    assert len(restaurant_indexed_reviews) == 0, "restaurant_indexed_reviews must be empty"
    for _, row in dataFrame.iterrows():
        u_temp = row.user_id
        i_temp = row.business_id
        rating = float(row.stars)
        if u_temp not in user_indexed_reviews:
            user_indexed_reviews[u_temp] = dict()
        user_indexed_reviews[u_temp][i_temp] = rating
        if i_temp not in restaurant_indexed_reviews:
            restaurant_indexed_reviews[i_temp] = dict()
        restaurant_indexed_reviews[i_temp][u_temp] = rating

def get_test_users(user_indexed_reviews, required_review_num):
    """
    give user_indexed_reviews dictionary and required_review_num, 
    return a set of user_id that only contains users who has more than required_review_num reviews
    """
    tested_uid = set()
    for user, reviews in user_indexed_reviews.items():
        if len(reviews) >= required_review_num:
            tested_uid.add(user)
    return tested_uid


def get_tests_and_update_reviews(user_indexed_reviews, restaurant_indexed_reviews, test_user_set, test_percentage):
    """
    update user/restaurant_indexed_reviews based on test_user_set, set aside test_percentage reviews from users in test_user_set for testing purposes
    return the dictionary that contains testing data
    """
    test_user_data = dict()
    for test_user in test_user_set:
        test_user_data[test_user] = dict()
        total_review_num = len(user_indexed_reviews[test_user])
        test_review_num = int(total_review_num * test_percentage)
        for i, (restaurant, rating) in enumerate(user_indexed_reviews[test_user].items()):
            if len(restaurant_indexed_reviews[restaurant]) == 1: # restaurant only has one review, don't delete it
                test_review_num += 1 # go to next item
                continue
            test_user_data[test_user][restaurant] = rating
            del user_indexed_reviews[test_user][restaurant]
            del restaurant_indexed_reviews[restaurant][test_user]
            if i == test_review_num:
                break 
    return test_user_data

def update_training_set(user_indexed_reviews, restaurant_indexed_reviews, training_percentage):
    for user, restaurant_reviews in user_indexed_reviews.items():
        total_reviews = len(restaurant_reviews)
        training_reviews = int(total_reviews * training_percentage)
        if training_reviews <= 0:
            training_reviews = 1 
        delete_reviews = total_reviews - training_reviews
        for i, (restaurant, reviews) in enumerate(restaurant_reviews.items()):
            if i >= delete_reviews:
                break
            del user_indexed_reviews[user][restaurant]
            del restaurant_indexed_reviews[restaurant][user]

def cal_average_rating(user_indexed_reviews):
    """
    for a given user, calculate this user's average rating for all the reviews in user_indexed_reviews table
    """
    total = 0.
    count = 0
    for user in user_indexed_reviews:
    	for restaurant, rating in user_indexed_reviews[user].items():
    		total += rating
    		count += 1
    return total/count


def main(argv):
    # set necessary parameters
	review_minimum_num = 30
	test_percentage = 0.2 # percentage of test data in all data set
	training_percentage = 0.9 # percentage of actual training set in all training data 
	data_size = 'Small'
	training_method = 'SVD' # random, CF, SVD, CBCF, WBCF
	pick_test_data = False
	savefile = False
	print "review_minimum_num:", review_minimum_num
	print "test_percentage:", test_percentage
	print "training_percentage:", training_percentage
	print "data_size:", data_size
	print "pick_test_data:", pick_test_data
	# initialize variable    
	#random evaluation
	if training_method == 'random':
	    print "calculating random rmse..."
	    random.seed()
	    random_evaluations = random_evaluating(test_user_data)
	    random_rmse = cal_rmse(random_evaluations)
	    print "final total CF rmse for the test data is:", random_rmse
	    return
	# other methods: initialize data set
	user_indexed_reviews = dict()  # user -> review
	restaurant_indexed_reviews = dict()  # {'business id': {'user':[review]}}, where review is a dict {'text':"It is good. "}
	smalldf = loadData.Parser()
	# build reviews that can be indexed from both user_id and restaurant_id 
	print "building indexed dictionaries..."
	build_user_and_restaurant_indexed_reviews(smalldf, user_indexed_reviews, restaurant_indexed_reviews)   
	print "setting data for test purposes..."
	test_user_set = get_test_users(user_indexed_reviews, review_minimum_num)
	#print "test_user_set number:",len(test_user_set)
	test_user_data = get_tests_and_update_reviews(user_indexed_reviews, restaurant_indexed_reviews, test_user_set, test_percentage)
	print "total number of users in test_user_data:", len(test_user_data)
	#update_training_set(user_indexed_reviews, restaurant_indexed_reviews, training_percentage)
	print "total number of users in training data:", len(user_indexed_reviews)
	print "total number of restaurants in training data:", len(restaurant_indexed_reviews)

	"""user_indexed_reviews: training dataset
	   test_user_data: test dataset
	"""
	base_evaluation = baseline.base_evaluating(test_user_data, user_indexed_reviews, restaurant_indexed_reviews)
	base_rmse = evaluation.calRMSE(base_evaluation)
	print "baseline rmse for test data is:", base_rmse

if __name__ == '__main__':
    main(sys.argv)
