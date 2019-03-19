'''
Q1_A 
KNN clustering
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

###looking on the dataset
dataset = make_blobs(n_features=2, centers=2) #make_blob output is a tuple data [0], labels [1] 
#Ploting 
colors = {1:'red', 0:'blue'}
plt.scatter(x = dataset[0][:,0],y = dataset[0][:,1], color = [colors[col] for col in dataset[1]])
#sb.scatterplot(x = dataset[0][:,0],y = dataset[0][:,1], hue = dataset[1]) #Seaborn way



###KNN function
def knn_classif(dataset, centroid=2, plots = True, plot_final = True):
    ## Random assign start
    index = np.random.choice(dataset[0].shape[0], centroid, replace = False) 
    #chose N index as centroids (make sure that not selet the same index)
    centroids = dataset[0][index] #assign N random centroids
    #Ploting 
    colors = {9:'coral',8:'tomato',7:'magenta',6:'aqua',5:'khaki',
              4:'cyan',3:'orange',2:'green', 1:'red', 0:'blue'}
    if plots:
        plt.scatter(x = dataset[0][:,0],y = dataset[0][:,1])
        plt.scatter(centroids[:,0], centroids[:,1], marker='*', s=200, c='k')
        plt.title('Initial Random centroids')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
     
    ## Assign observation to the nearest mean
    #empty matrix (row is an observation, column the number of centroids)
    distances = np.zeros((dataset[0].shape[0],centroid))
    #calculate distances for all values to centroid
    for ncent in range(0,centroid):
        distn = np.sqrt(np.sum((dataset[0]-centroids[ncent])**2,axis = 1))
        distances[:,ncent] = distn
    #assign labels according to initial distances
    labels = np.argmin(distances, axis=1)
    if plots:
        plt.scatter(x = dataset[0][:,0],y = dataset[0][:,1], color = [colors[col] for col in labels], alpha= 0.4)
        plt.scatter(centroids[:,0], centroids[:,1], marker='*', s=200, c=[colors[col] for col in range(0,centroid)])
        plt.title('First labels assigned')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
    #recalculate centroid using labels (mean x, mean y)
    for cent in range(0,centroid):
        centroids[cent] = dataset[0][labels == cent].mean(axis = 0) #it calculates the centroid for each labeled class
    #Ploting  new initial centroids.
    if plots:
        plt.scatter(x = dataset[0][:,0],y = dataset[0][:,1], color = [colors[col] for col in labels], alpha= 0.4)
        plt.scatter(centroids[:,0], centroids[:,1], marker='*', s=200, c=[colors[col] for col in range(0,centroid)])
        plt.title('Centroids initialized')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
    #This is going to be our initial condition. The loop is going to recalculate
    #new centrodis, and is going to re-label the datapoints until it converges.
    
    #calculate distance difference between previous centroid
    old_centroids = np.zeros(centroids.shape)
    cent_dist = np.linalg.norm(centroids - old_centroids, axis=None) # axis None output is only a number
    iteration = 0
    while cent_dist != 0: #threshold to stop the loop (convergence of centroids)
        old_centroids = np.copy(centroids) # pass the old centroids
        
        for ncent in range(0,centroid): #calculate distance over centroids
            distn = np.sqrt(np.sum((dataset[0]-centroids[ncent])**2,axis = 1))
            distances[:,ncent] = distn
        labels = np.argmin(distances, axis=1)
        #calculate new centroid
        for cent in range(0,centroid):
            centroids[cent] = dataset[0][labels == cent].mean(axis = 0) 
        iteration += 1
        cent_dist = np.linalg.norm(centroids - old_centroids, axis=None)
        if iteration > 500: break
    #Ploting  final clusters.
    if plot_final:
        plt.scatter(x = dataset[0][:,0],y = dataset[0][:,1], color = [colors[col] for col in labels], alpha= 0.4)
        plt.scatter(centroids[:,0], centroids[:,1], marker='*', s=200, c=[colors[col] for col in range(0,centroid)])    
        plt.title('Final clusters for K = %d' % centroid)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
        print('Number of iterations: %d' % iteration)
    #calculate SSE
    SSE = 0    
    for K in range(0,centroid):
        l2 = np.linalg.norm((dataset[0][labels == K] - centroids[K]), axis=1)**2
        SSE = SSE + l2.sum()
    print('SSE for K-%d = %.2f' % (centroid,SSE))
    return SSE

#Plotting SSE for different K_values
dataset = make_blobs(n_features=2, centers=2)
all_SSE = []
for k in range(1,11):
    all_SSE.append(knn_classif(dataset, centroid=k,plot_final=False, plots = False))
plt.plot(range(1,11),all_SSE)
plt.xticks(np.arange(10))
plt.xlabel('K values')
plt.ylabel('SSE')
plt.title('SSE~K-values for 2 cluster centers')
plt.show()

dataset = make_blobs(n_features=2, centers=5)
all_SSE = []
for k in range(1,11):
    all_SSE.append(knn_classif(dataset, centroid=k, plots = False, plot_final=False))
plt.plot(range(1,11),all_SSE)
plt.xticks(np.arange(10))
plt.xlabel('K values')
plt.ylabel('SSE')
plt.title('SSE~K-values for 5 cluster centers')
plt.show()




