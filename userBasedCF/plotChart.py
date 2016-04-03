import matplotlib.pyplot as plt

#k = [20,25,30,35,40,45,50]

def plotChart(k,v,label,yLabel):
    plt.plot(k,v, marker='*', color='b', linewidth=2.0, label=label)
    plt.xlabel('value of k')
    plt.ylabel(yLabel)
    plt.legend()
    plt.show()


#plt.plot(k,v2, marker='o', linestyle='--', color='r', linewidth=2.0,label='userBased')


def plotTwoChart(k,v1,v2,label1,label2,yLabel):
    plt.plot(k,v1, marker='*', color='b', linewidth=2.0, label=label1)
    plt.plot(k,v2, marker='o', linestyle='--', color='r', linewidth=2.0,label=label2)
    plt.xlabel('value of k')
    plt.ylabel(yLabel)
    plt.legend()
    plt.show()