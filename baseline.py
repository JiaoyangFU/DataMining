import pandas as pd
import numpy as np
import scipy.sparse as sps
from itertools import izip
import csv

dataFrame = pd.read_csv('inputData2.csv',sep='\t')
newdf = dataFrame[['user_id','business_id','stars']]

u_size = newdf.user_id.nunique()
i_size = newdf.business_id.nunique()

# Dict to store ID mapping info.
users_id_map = {}
items_id_map = {}

# Converting user_id and item_id to make them continuous starting from 0
u_id = 0  # New id for users starting from 0 to size-1
i_id = 0  # New id for items starting from 0 to size-1


# As we read the data, we will perform the division for k folds.
# And ratings matrices for all folds and outside of folds will be
# stored in lists of matrices.

rating_dict = {}  # dict to store ratings
for _, row in newdf.iterrows():
    u_temp = row.user_id
    i_temp = row.business_id
    if u_temp not in users_id_map:
        users_id_map[u_temp] = u_id
        u_id += 1
    if i_temp not in items_id_map:
        items_id_map[i_temp] = i_id
        i_id += 1
    uu = users_id_map[u_temp]
    ii = items_id_map[i_temp]
    pair = (uu, ii)
    rating = float(row.stars)
    if pair not in rating_dict:
        rating_dict[pair] = [rating]
    else:
        rating_dict[pair].append(rating)


w = csv.writer(open("rating.csv", "w"))
for key, val in rating_dict.items():
    w.writerow([key, val])
'''
calculate the rating from reviews, if a user has multiple reviews for a restaurant, use the average rating
'''
for pair in rating_dict:
    avg_rating = np.mean(rating_dict[pair])
    rating_dict[pair] = avg_rating

avg_star = dataFrame['stars'].mean()

user_avg = newdf.groupby(['user_id']).mean()
biz_avg = newdf.groupby(['business_id']).mean()
user_avg.reset_index(level=0, inplace=True)
biz_avg.reset_index(level=0, inplace=True)

#baseline = sps.dok_matrix((u_size, i_size))
base_dict = {}
for user_id, stars in izip(user_avg['user_id'],user_avg['stars']):
    uid = users_id_map[user_id]
    ustars = float(stars)
    for business_id,stars in izip(biz_avg['business_id'],biz_avg['stars']):
        bid = items_id_map[business_id]
        bstars = float(stars)
        pair = (uid, bid) 
        if pair in rating_dict:
            base_dict[pair] = ((avg_star + (bstars - avg_star) + (ustars - avg_star)), rating_dict[pair])       
        #baseline[uid,bid] = bstars + ustars - avg_star
        

w = csv.writer(open("baseline.csv", "w"))
for key, val in base_dict.items():
    w.writerow([key, val])

