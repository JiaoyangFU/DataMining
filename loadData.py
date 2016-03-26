import json
import pandas as pd
import numpy as np


def Parser():
	filepath = '/Users/Stella/Documents/yelp/data/yelp_training_set_'
	#filepath = '/Users/Stella/Documents/yelp/data/yelp_academic_dataset_'
	with open(filepath + 'review.json') as f:
	    reviews = pd.DataFrame(json.loads(line) for line in f)

	with open(filepath + 'user.json') as f:
	    users = pd.DataFrame(json.loads(line) for line in f)

	with open(filepath + 'business.json') as f:
	    business = pd.DataFrame(json.loads(line) for line in f)

	ind = []
	for i in range(len(business.categories)):
		if 'Restaurants' in business[i:i+1].categories.to_string():
			ind.append(i)
	restaurant = business.iloc[ind]

	business1 = restaurant[restaurant.city == "Phoenix"][['business_id', 'name', 'stars', 'review_count', 'city','categories']]
	#business1 = restaurant[['business_id', 'name', 'stars', 'review_count', 'city','categories']]
	#business1 = restaurant[(business.city == "Las Vegas")][['business_id', 'name', 'stars', 'review_count', 'city','categories']]
	reviews1 = reviews[['business_id', 'user_id', 'stars', 'review_id']]
	users1 = users[['user_id', 'name', 'review_count', 'average_stars']]
	data = pd.merge(business1, reviews1, on = 'business_id', how = 'inner')
	data.head(2)
	dataFrame = pd.merge(data, users1, on = 'user_id', how = 'inner')
	#dataFrame = dataFrame.sort_values(by = 'business_id')
	new_col = dataFrame.columns.values
	new_col[1] = 'business_name'
	new_col[2] = 'business_avg'
	new_col[3] = 'business_review_count'
	new_col[7] = 'stars'
	new_col[9] = 'user_name'
	new_col[10] ='user_review_count'
	new_col[11] = 'user_avg'
	dataFrame.columns = new_col

	dataFrame.to_csv('inputData2.csv', sep='\t', encoding='utf-8')

	small = recompute_frame(dataFrame)
	small = small[(small.business_review_count > 50)]
	smalldf = recompute_frame(small)
	smalldf.to_csv('small.csv',  sep='\t', encoding='utf-8')

	return smalldf

def recompute_frame(ldf):
    ldfu=ldf.groupby('user_id')
    ldfb=ldf.groupby('business_id')
    user_avg=ldfu.stars.mean()
    user_review_count=ldfu.review_id.count()
    business_avg=ldfb.stars.mean()
    business_review_count=ldfb.review_id.count()
    nldf=ldf.copy()
    nldf.set_index(['business_id'], inplace=True)
    nldf['business_avg']=business_avg
    nldf['business_review_count']=business_review_count
    nldf.reset_index(inplace=True)
    nldf.set_index(['user_id'], inplace=True)
    nldf['user_avg']=user_avg
    nldf['user_review_count']=user_review_count
    nldf.reset_index(inplace=True)
    return nldf

def explData():
	from matplotlib import pyplot as plt
	dataFrame = pd.read_csv('inputData2.csv',sep='\t')
	urc=dataFrame.groupby('user_id').review_id.count()
	ax=urc.hist(bins=50, log=True)
	plt.xlabel("Reviews per user")
	plt.title("Review Count per User")
	plt.savefig('Review Count per User2.png')
	plt.show()

	brc=dataFrame.groupby('business_id').review_id.count()
	ax=brc.hist(bins=50, log=True)
	plt.xlabel("Reviews per restaurant")
	plt.title("Review Count per Restaurant")
	plt.savefig('Review Count per Restaurant2.png')


	ax=dataFrame.stars.hist(bins=5)
	plt.xlabel("Star rating")
	plt.title("Star ratings over all reviews");
	plt.savefig('Overall Star Rating2.png')

	urc = dataFrame.groupby('city').review_id.count()
	urc.columns = ['city','count']
	urc.sort_values(by = 'count', ascending = False)
	city = urc.head(10)
	colors  = ["coral","blue", "yellow","green","pink"]
	city.plot(kind='pie',label='city',colors=colors,explode=(0.1, 0, 0, 0, 0),autopct='%1.1f%%', shadow=True)
	plt.title("Top five cities");
	plt.savefig('Top five cities2.png')

