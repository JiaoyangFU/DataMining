import matplotlib.pyplot as plt

k = [5,10,20,50,100,150]
v1 = [1.78145842838, 1.63819403814, 1.36941798713,1.1104912017, 1.04673785959, 1.04322825085]
v2 = [1.98145842838, 1.73819403814, 1.46941798713,1.1104912017, 1.05673785959, 1.14322825085]


plt.plot(k,v1, marker='*', color='b', linewidth=2.0, label='ItemBased')
plt.plot(k,v2, marker='o', linestyle='--', color='r', linewidth=2.0,label='user')
plt.xlabel('value of k')
plt.ylabel('RMSE value')
plt.legend()
plt.show()