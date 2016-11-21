#!/usr/bin/python

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


'''
Check wikipedia for more information:
https://en.wikipedia.org/wiki/K-means_clustering

Challenge: To do it in a pure pythonic manner

Algorithm:
    1.) Initialized random "means"

    2.) Compute Euclidean distance from means of each point

    3.) Assign clusters using min distance

    4.) Compute new means of clusters

    5.) Go to back to 2 until there is no change in cluster assignments

    Note: We should handle cases when a cluster is an empty list in the
    algorithm. People say to just go back to 1 online, but that seems very brute
    to me.

'''


def distance(a,b):
    '''

    sum(sqrt((x[i] - mean[i])^2))

    '''
    term = []
    for i in range(len(a)):
        term.append(math.pow(a[i] - b[i],2))
    ret = math.sqrt(sum(term))
    return ret

def average(iarr):
    '''

    Finding new mean in each cluster.
    For each cluster:
        [sum(x)/N, sum(y)/N, sum(z)/N... etc]
    i is your coordinate(x,y,z....etc)

    '''
    av = []
    for cluster in iarr:
        columns = []
        for i in range(len(cluster[0])):
            columns.append([row[i] for row in cluster])
        mean = [sum(columns[i])/len(columns[i]) for i in range(len(columns))]

        av.append(mean)
    return av

def kmeans(data, nclusters, rscale):

    '''

    data : [ [xi], [yi], [zi]...]

    nclusters : should rename to nclusters

    rscale : Just a scaling of the random val between [0,1)

    return cluster:
        [ [ [xi, yi, zi... ], [xi+1, yi+1, zi+1...]... ], ... ncluster lists]

    '''
    temp = []
    for n in range(len(data[0])):
        temp2 = []
        for i in range(len(data)):
            temp2.append(data[i][n])
        temp.append(temp2)

    Points = temp

    temp = [] # [ [xi,yi,zi...] ]
    for n in range(nclusters):
        temp2 = [] # x,y,z...
        for i in range(len(data)):
            temp2.append(random.random()*rscale)
        temp.append(temp2)

    means = temp

    oldcluster = [[0 for i in range(len(data))] for i in range(nclusters)]
    cluster = [ [] for i in range(nclusters)]
    while True:
        for i in range(len(Points)):

            d = [math.pow(distance(Points[i], means[n]),2) for n in range(nclusters)]
            mini = 0
            do = d[0]

            for z in range(len(d)):
                if do > d[z]:
                    do = d[z]
                    mini = z

            cluster[mini].append(Points[i])

        if cluster == oldcluster:
            print("Converged")
            break

        newmean =  average(cluster)

        oldcluster = cluster
        cluster = [ [] for i in range(nclusters)]
        means = newmean
    return cluster



if __name__ == "__main__":

    x = []
    w = []
    y = []
    z = []
    rscale = 10
    clusters = 10

    for i in range(10000):
        x.append(random.random()*rscale)
        y.append(random.random()*rscale)
        z.append(random.random()*rscale)
        w.append(random.random()*rscale)

    data = [x, y, z]
    clusters = kmeans(data, 10, rscale)
    print(len(clusters))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for c in clusters:
        ax.plot(np.array(c)[:,0],np.array(c)[:,1],np.array(c)[:,2], "*")
    plt.show()

