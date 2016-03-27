# "	user_id	 business_id	business_name	business_avg	business_review_count
#   city	 categories	stars	review_id	user_name	user_review_count	user_avg"

import operator
import math
import pandas as pd
import numpy as np
import scipy.sparse as sps
from itertools import izip
import csv
from evaluation import calRMSE
from evaluation import calMAE

''' 
define global variable  

'''

usersIdMap = {} # map user_id to an integer
itemsIdMap = {} # map business_id to an integer
userItemsMap = {} # store all items' id the given user rated => {mapped user_id : [itemId1, itemId2, itemId3,...]}
userRatingMean = {} # store the average rating value for the given user => {mapped user_id : average rating}
ratingDict = {}  # store ratings=>{pair(mapped user_id,mapped bussiness_id) : stars for the given two id}

CosineSimilarityMap = {} # store similarity between two users =>{(mapped user_id1,mapped user_id2), similarity}
userNeighborMap = {} # store the Neighbors list for the given user in decending order of their similarity => {mapped user_id : [id1,id2,id3.....]}
predictionMap = {} # store prediction for uer u's preference for an item i= > {(userId,itemId): prediction}
predictionRealMap = {} # {(uid, bid):(predict_value, true_value)}

dataFrame = pd.read_csv('../../small.csv',sep='\t')
newdf = dataFrame[['user_id','business_id','stars','user_avg']]

userCnt=newdf.user_id.nunique() # the totoal number of users => 495
itemCnt=newdf.business_id.nunique() # the totoal number of items => 281
    
#RMSE = calRMSE({})
#MAE =calMAE({})

'''
    this function is used to store the given map as .cvs file
    
'''

def storeContents(mapName,fileName):
    fileName=fileName+ ".csv"
    w = csv.writer(open(fileName, "w"))
    for key, val in mapName.items():
        w.writerow([key, val])

# function  storeRatingDict() END


'''
    initialize usersIdMap,  itemsIdMap, ratingDict, userItemsMap    userRatingMean,
    
'''
def initialization():
    id1=0
    id2=0
    for index, row in newdf.iterrows(): # iterrows gives you (index, row) tuples rather than just the rows
        userId=row.user_id
        itemId=row.business_id
        if userId not in usersIdMap:
            usersIdMap[userId]=id1
            id1+=1

        if itemId not in itemsIdMap:
            itemsIdMap[itemId]=id2
            id2+=1
    
        mappedUserId=usersIdMap[userId]
        mappedItemId=itemsIdMap[itemId]
        
        if mappedUserId not in userItemsMap:
            userItemsMap[mappedUserId] = [mappedItemId]
        else:
            userItemsMap[mappedUserId].append(mappedItemId)

        if mappedUserId not in userRatingMean:
            userRatingMean[mappedUserId] =row.user_avg


        pair = (mappedUserId,mappedItemId)
        if pair not in ratingDict:
            ratingDict[pair] = float(row.stars)
    #end for

# function void initialization() END


'''
    calculate Cosin Similarity for the given two users userId1, userId2
    
'''
def getCosineSimilarity(userId1, userId2): # userId1, userId2 are mapped user id
    sumxx,sumyy,sumxy=0.0, 0.0, 0.0;
    itemsList1=userItemsMap[userId1]
    itemsList2=userItemsMap[userId2]

    for itemId1 in itemsList1:
        x=ratingDict[(userId1,itemId1)]
        sumxx += x * x
        if itemId1 in itemsList2:
            y=ratingDict[(userId2,itemId1)]
            sumxy += x * y


    for itemId2 in itemsList2:
        y=ratingDict[(userId2,itemId2)]
        sumyy += y * y
    #print "sumxy= " + str(sumxy) +  "sumxx= " + str(sumxx)+"sumyy= " + str(sumyy)
    if abs(sumxx) <=0.000000001 or  abs(sumyy) <=0.000000001 or abs(sumxy) <=0.000000001:
        return 0

    return sumxy/(math.sqrt(sumxx) * math.sqrt(sumyy))
# getCosineSimilarity(userId1, userId2) END


'''
    calculate Cosin Similarity for all users
    
'''
def calculateCosineSimilarityMap():
    for id1 in range(userCnt): # from 0 to userCnt-1
        CosineSimilarityMap[(id1,id1)]=1
        for id2 in range(id1+1,userCnt): # from id1 to userCnt-1
            similarity =getCosineSimilarity(id1, id2)
            CosineSimilarityMap[(id1,id2)]=similarity
            CosineSimilarityMap[(id2,id1)]=similarity

# getCosineSimilarityMap() END


'''
    sort the neighbors of a given user in decending order of their similarity and store in userNeighborMap
    users do not consider  themselvse as neighbor to themselvse
    
'''
def sortNeighbors():
    similarityMap = {} # similarity map for a given user
    tupleList =[] # [((id1,id2),similarity), (),()..] => a list of tuples in decending order of similarity
    for id1 in range(userCnt): # from 0 to userCnt-1
        for id2 in range(userCnt): # from id1 to userCnt-1
            if id1 == id2:
                continue
            similarityMap[(id1,id2)]= CosineSimilarityMap[(id1,id2)]
        #END for
        # return  a list of tuples sorted by the second element (similarity) in each tuple in decreasing order
        tupleList=sorted(similarityMap.items(),key=operator.itemgetter(1),reverse=True)
        for i in range(len(tupleList)):
            neighborId=tupleList[i][0][1]
            if id1 not in userNeighborMap:
                userNeighborMap[id1]=[neighborId]
            else:
                userNeighborMap[id1].append(neighborId)
        #END for
    #END for

# function sortNeighbors() END

'''
    computing and store prediction for user u's preference for an item i
    k is the number of neighbors will be checked whether qualifed to considered in computing the prediction
    similarityThreshold is  threshold of similarity for neighbors that can be considered
    
'''
def computingPrediction(k,similarityThreshold):
    for userId  in range(userCnt): # from 0 to userCnt-1
        for itemId in range(itemCnt): # from id1 to itemCnt-1
            similaritySum=0.0 # float value
            weightedAverageSum=0.0
            cnt= min(k,len (userNeighborMap[userId]))
            for i in range(cnt):
                uId=userNeighborMap[userId][i]
                if itemId not in userItemsMap[uId]: # uId didn't rate itemId before
                    continue
                
                similarity=CosineSimilarityMap[(userId,uId)]
                if abs(similarity-similarityThreshold)<=0.00000001:
                    continue

                similaritySum += similarity
                weightedAverageSum += similarity * ( ratingDict[(uId,itemId)]- userRatingMean[uId])

        #END for
        if abs(similaritySum-0) <= 0.00000001:
            predictionMap[(userId,itemId)]=userRatingMean[userId]
        else:
            predictionMap[(userId,itemId)]=userRatingMean[userId] + weightedAverageSum/similaritySum
            
        if (userId,itemId) in ratingDict:
            predictionRealMap[(userId,itemId)] =(predictionMap[(userId,itemId)], ratingDict[(userId,itemId)])
        else: # userId didn't rate itemId before, so the original value is considered as 0
            predictionRealMap[(userId,itemId)] =(predictionMap[(userId,itemId)], 0)
    #END for

# function computingPrediction(k,similarityThreshold) END

def main():
    evaluationMap ={} # {k: [RMSE,MAE]}
    
    initialization()
    #storeContents(ratingDict,"ratingDict")
    
    calculateCosineSimilarityMap()
    #storeContents(CosineSimilarityMap,"CosineSimilarityMap")
    
    sortNeighbors()
    #storeContents(userNeighborMap,"userNeighborMap")
    
    similarityThreshold=0
    stepList=[5,10,20,50,100,150]
    for k in stepList:
        computingPrediction(k,similarityThreshold)
        #fileName1="userBasedCFRating" + str(k)
        #storeContents(predictionMap,fileName1)
        
        RMSE = calRMSE(predictionRealMap)
        MAE =calMAE(predictionRealMap)
        evaluationMap[k]=[RMSE,MAE]
    
    storeContents(evaluationMap,"evaluationMap1")

# function main END


main()


'''
    usersIdMap = {} # map user_id to an integer
    itemsIdMap = {} # map business_id to an integer
    userItemsMap = {} # store all items' id the given user rated => {mapped user_id : [itemId1, itemId2, itemId3,...]}
    userRatingMean = {} # store the average rating value for the given user => {mapped user_id : average rating}
    ratingDict = {}  # store ratings=>{pair(mapped user_id,mapped bussiness_id) : stars for the given two id}
    
    CosineSimilarityMap = {} # store similarity between two users =>{(mapped user_id1,mapped user_id2), similarity}
    userNeighborMap = {} # store the Neighbors list for the given user in decending order of their similarity => {mapped user_id : [id1,id2,id3.....]}
    predictionMap = {} # store prediction for uer u's preference for an item i= > {(userId,itemId): prediction}

'''

#storeContents(userItemsMap,"userItemsMap")
#storeContents(userRatingMean,"userRatingMean")
#storeContents(CosineSimilarityMap,"CosineSimilarityMap")
#storeContents(userNeighborMap,"userNeighborMap")
































