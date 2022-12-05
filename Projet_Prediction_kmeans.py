# -*- coding: utf-8 -*-
"""
Projet Machine learning, prediction methods and applications

Groupe A:
    AYI KPADONOU Aldah 
    BONOU Norgile
    AMOUSSOU Bourgeot

"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(123)
#%%

"""
Exercise: K-means from scratch

In this exercise, you will code an algorithm from scratch to perform K-means clustering.

1) Data generation
"""

def clust1 (s, T):
    """
    Generate a Cluster 1

    Parameters
    ----------
    s : TYPE
        DESCRIPTION. Standard deviation
    T : Size of sample

    Returns
    -------
    Two feactures x_1 and x_2

    """
    
    x1 = np.random.normal(1,s, size = T)
    x2 = np.random.normal(1,s, size = T)
    
    return x1, x2

def clust2 (s, T):
    """
    Generate a Cluster 2

    Parameters
    ----------
   
    s : TYPE
        DESCRIPTION. Standard deviation
    T : Size of sample

    Returns
    -------
    Two feactures x_1 and x_2

    """
    
    x1 = np.random.normal(1,s, size = T)
    x2 = np.random.normal(-1,s, size = T)
    
    return x1, x2

def clust3 (s, T):
    """
    Generate a Cluster 3

    Parameters
    ----------
    
    s : TYPE
        DESCRIPTION. Standard deviation
    T : Size of sample

    Returns
    -------
    Two feactures x_1 and x_2

    """
    
    x1 = np.random.normal(-1,s, size = T)
    x2 = np.random.normal(-1,s, size = T)
    
    return x1, x2

def clust4 (s, T):
    """
    Generate a Cluster 4

    Parameters
    ----------
    m : TYPE
        DESCRIPTION : Is the mean of different cluster. In our case we have two 
        different list of value for m as follows : m1 = [1,1,-1,-1], m2= [1,-1,-1,1]
    s : TYPE
        DESCRIPTION. Standard deviation
    T : Size of sample

    Returns
    -------
    Two feactures x_1 and x_2

    """
    
    x1 = np.random.normal(-1,s, size = T)
    x2 = np.random.normal(1,s, size = T)
    
    return x1, x2
#%%
"""
2. According to the previous data simulation model, setting σ = 0.1, generate K = 4 clusters, each cluster
should contain 25 observations. Draw the corresponding scatterplot, using a different color for each
cluster. Draw the same scatterplot for different values of σ (σ = 0.2, 0.3, 0.4, 0.5, ...). Explain the effect
of parameter σ on the resulting clusters.
"""
def cluster(sigma,T):
    """
    Generate different cluster for different value of sigma

    Parameters
    ----------
    sigma : TYPE : float
        DESCRIPTION : Standard deviation
    T : TYPE : integer
        DESCRIPTION :  size of sample

    Returns
    -------
    cluster1 : TYPE : List
        DESCRIPTION : Contain list of array for different cluster
    cluster2 : TYPE : List
        DESCRIPTION : Contain list of array for different cluster
    cluster3 : TYPE : List
        DESCRIPTION : Contain list of array for different cluster
    cluster4 : TYPE : List
        DESCRIPTION : Contain list of array for different cluster

    """
    cluster1 = []
    cluster2 = []
    cluster3 = []
    cluster4 = []
    
    for i in range(len(sigma)):
        
        cluster1.append(clust1(sigma[i], T))
        cluster2.append(clust2(sigma[i], T))
        cluster3.append(clust3(sigma[i], T))
        cluster4.append(clust4(sigma[i], T))
        
    return cluster1, cluster2, cluster3, cluster4

#Define the value of sigma
s = [round(x,1) for x in np.arange(0.1,1.0,0.1)]

#Call the function and store the results in different variable
cluster1, cluster2, cluster3, cluster4 = cluster(sigma = s ,T=25)


#Plot of each cluster according the specific value of sigma

for i in range(len(s)):
    
    sns.set_theme(style="darkgrid")
    plt.figure(figsize = [12,9])
    sns.scatterplot(x =cluster1[i][0], y=cluster1[i][1])
    sns.scatterplot(x =cluster2[i][0], y=cluster2[i][1])
    sns.scatterplot(x =cluster3[i][0], y=cluster3[i][1])
    sns.scatterplot(x =cluster4[i][0], y=cluster4[i][1])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Scatterplot for sigma = {} ".format(s[i]))
    plt.legend(["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4"])
    plt.show()

"""
The effect of parameter sigma on the resulting clusters.

Standard Deviation is a measure of how much the data is dispersed from its mean. 
A high standard deviation implies that the data values are more spread out from the mean

Regarding the plots, we find that as the sigma value increases, there is a significant 
dispersion of observations. That is to say that the observations are scattered more and more on the graph.

We keep for the next question sigma = 0.1
"""

#%%
"""
3. The training dataset should be a matrix X ∈ Rn×2(n = 25K observations and 2 features). Propose a
way to initialize K-means, in other words: randomly pick K cluster centroids µ1, µ2, ..., µK.


"""
#Create a dataset by combining all the clusters by using sigma = 0.1
data = np.zeros((100,2))
data[0:25,0], data[0:25,1] = cluster1[0][0],  cluster1[0][1]
data[25:50,0], data[25:50,1] = cluster2[0][0],  cluster2[0][1]
data[50:75,0], data[50:75,1] = cluster3[0][0],  cluster3[0][1]
data[75:100,0], data[75:100,1] = cluster4[0][0],  cluster4[0][1]  

np.random.seed(123)

#Define a function to randomly pick K cluster centroids
def init_cent(data, K):
    """
    Randomly pick K cluster centroids

    Parameters
    ----------
    data : TYPE : array nxk
        DESCRIPTION : Dataset
    K : TYPE : integer
        DESCRIPTION : Initialize the number of cluster

    Returns
    -------
    centr : TYPE : Array
        DESCRIPTION : Cluster centroids

    """
    n,k = data.shape
    idx= np.random.choice(n, K,replace=False)
    centr=data[idx,:]
    
    return centr

#Call the function and store the results in variable
Cluster_centroids = init_cent(data, K=4)

#%%
"""
4. Implement the cluster assignment step: for each observation x
(i), assign this observations to its closest cluster centroid, 
in other words compute for all i ∈ {1, ..., n}:
"""
def clust_assignment(data,centroids,K):
    """
    Closest cluster for each observations.

    Parameters
    ----------
    data : TYPE : array
        DESCRIPTION Dataset
    centroids : TYPE : array
        DESCRIPTION. Cluster centroid
    K : TYPE : integer
        DESCRIPTION : Number of cluster

    Returns
    -------
    c : TYPE : Distance between each observation and centroid
        DESCRIPTION.
    min_c_index : TYPE : array
        DESCRIPTION : Index of each minimum distances

    """
    
    n,k = data.shape
    
    c = np.zeros((n,K))
    
    for j in range(K):
        for i in range(n):
            
            c[i,j] = (data[i][0]- centroids[j][0])**2 + (data[i][1]- centroids[j][1])**2 
    
    
    min_c_index = np.array([np.argmin(i) for i in c ])
    
    return c, min_c_index

distances, closest_cluster = clust_assignment(data, centroids = Cluster_centroids, K=4)

            
#%%
"""
5. Implement the update cluster centroid step: for each cluster k, consider the set of obervations assigned
to cluster k, and compute the average (mean) of these observations, in other words, compute for all
k ∈ {1, ..., K}
"""
def mean_centr(data, closest_cluster, K):
    """
    Mean centroid

    Parameters
    ----------
    data : TYPE : array
        DESCRIPTION. Dataset
    closest_cluster : TYPE : array
        DESCRIPTION. Closest cluster for each observations
    K : TYPE : integer
        DESCRIPTION : Number of cluster

    Returns
    -------
    mean_centroid : TYPE : array
        DESCRIPTION : Centroid

    """
    
    mean_centroid = [] 
    
    for j in range(K):
        mean_centroid.append(data[closest_cluster==j].mean(axis=0))
        
    return mean_centroid

mean_centroid = mean_centr(data, closest_cluster, K=4)

#%%
"""
6. Draw again the corresponding scatterplot, using a different color for each point in a cluster, and add
K points representing the current cluster centroids (you should use a different symbol to distinguish
between points and cluster centroids).
"""

u = np.unique(closest_cluster)

sns.set_theme(style="darkgrid")
plt.figure(figsize = [14,9])
for i in u:
    plt.scatter(data[closest_cluster == i , 0] , data[closest_cluster == i , 1] , label = i)
plt.scatter(mean_centroid[0][0],mean_centroid[0][1], marker = 'o', s=500, color = "b")
plt.scatter(mean_centroid[1][0],mean_centroid[1][1], marker = 'o',s=500, color = "orange")
plt.scatter(mean_centroid[2][0],mean_centroid[2][1], marker = 'o',s=500, color = "g")
plt.scatter(mean_centroid[3][0],mean_centroid[3][1], marker = 'o',s=500, color = "r")
plt.title("")
plt.legend()
plt.show()

#%%


#%%
"""
7. Use a loop in order to repeat steps 4 to 6. The number of iterations should be large enough for the
algorithm to have time to converge. 

Here we choose the number of iteration and add some convergence condition in order to get a better value 
of cluster centroid.
"""

valu_max = 10000
tol = 0.9


for it in range(valu_max):
    
    cent_1 = init_cent(data, K=4)
    
    _, min_c_1 = clust_assignment(data, centroids=cent_1, K=4)
    
    new_centro=mean_centr(data, closest_cluster = min_c_1, K=4)
    new_centro_def = pd.DataFrame(new_centro)
    new_centro_a = np.array(new_centro_def)
        
        
    if it:
        norm = np.linalg.norm(next_centroid - new_centro_a)
        if norm <tol:
            u = np.unique(min_c_1)
    
            sns.set_theme(style="darkgrid")
            plt.figure(figsize = [14,9])
            for i in  u: 
                plt.scatter(data[min_c_1 == i , 0] , data[min_c_1 == i , 1] , label = i)
            plt.scatter(new_centro[0][0],new_centro[0][1], marker = 'o', s=500, color = "b")
            plt.scatter(new_centro[1][0],new_centro[1][1], marker = 'o',s=500, color = "orange")
            plt.scatter(new_centro[2][0],new_centro[2][1], marker = 'o',s=500, color = "g")
            plt.scatter(new_centro[3][0],new_centro[3][1], marker = 'o',s=500, color = "r")
            plt.title("")
            plt.legend()
            plt.show()
            print("The value of iteration where the condition has been fulfilled", it+1)
            
                
                
            break
        
    next_centroid=new_centro_a
        
print(new_centro_a)
    
#%%
'''
8-Implement the whole algorithm in a same Python (or R) funtion, named Kmeans, that takes the matrix
X as input, and returns as outputs: the clusters assigned to the observations c
(1), c(2), ..., c(n) and the
cluster centroids µ1, µ2, ..., µK. Ideally, the function should plot the final scatterplot, with a different
color for each cluster, and the corresponding cluster centroids
'''
def Kmeans(data,K,valu_max,tol, random_state = 123):
    
    np.random.seed(random_state)
       
    for it in range(valu_max):
        cent_1 = init_cent(data, K=4)
        _, min_c_1 = clust_assignment(data, cent_1,K)

        new_centro=mean_centr(data, min_c_1, K)
        new_centro_def=pd.DataFrame(new_centro)
        new_centro_a=np.array(new_centro_def)
        
        
        if it:
            norm=np.linalg.norm(next_centroid - new_centro_a)
            if norm < tol:
                u_labels = np.unique(min_c_1)

                sns.set_theme(style="darkgrid")
                plt.figure(figsize = [14,9])
                for i in  u_labels: 
                    plt.scatter(data[min_c_1 == i , 0] , data[min_c_1 == i , 1] , label = i)
                plt.scatter(new_centro[0][0],new_centro[0][1], marker = 'o', s=500, color = "b")
                plt.scatter(new_centro[1][0],new_centro[1][1], marker = 'o',s=500, color = "orange")
                plt.scatter(new_centro[2][0],new_centro[2][1], marker = 'o',s=500, color = "g")
                plt.scatter(new_centro[3][0],new_centro[3][1], marker = 'o',s=500, color = "r")
                plt.title("")
                plt.legend()
                plt.show()
                print(it)
                break
           
        next_centroid=new_centro_a

        
        a=plt.show()
        it=it+1
    return min_c_1 , new_centro, a
    
cluster_label_8, cluster_centroid_8, _ = Kmeans(data,K=4, valu_max = 1000,tol = 0.9)


#%%
"""
9- Modify your function and implement “repeated” K-means, in order to reduce the risk of finding local
optima. The final function must be applied to the simulated data (see question 1), with different values
of σ. Propose a way to compute the quality of the obtained partition: is there a way to compute a
misclassification error rate in this case"""

def Kmeans(data,K ,valu_max,tol, random_state = 123):
    
    np.random.seed(random_state)
    
    n,k = data.shape
    
    
    all_cost_function=[]
    all_closest_cluster = []
    all_centroid = []
    for e in range(100):
        
        for it in range(valu_max):
            cent_1 = init_cent(data, K)
            _, min_c_1 = clust_assignment(data, cent_1, K)
        
            new_centro=mean_centr(data, min_c_1, K)
            new_centro_def = pd.DataFrame(new_centro)
            new_centro_a = np.array(new_centro_def)
            
            #K=new_centro.shape[0]
            if it:
                norm = np.linalg.norm(next_centroid - new_centro_a)
                if norm <tol:
                    
                    min_c = min_c_1  
                    
                    break
                
            next_centroid=new_centro_a
            
        J = np.zeros((n,data.shape[1]))
        cost_function = np.zeros((n,1))             
        for i in range(n):
                    
            for j in range(K):
                        
                if min_c_1[i]==j:
                            
                    J[i,:] = new_centro_a[j]
            cost_function[i] = (data[i][0]- J[i][0])**2 + (data[i][1]- J[i][1])**2
        all_cost_function.append(np.mean(cost_function))
         
        all_closest_cluster.append(min_c_1)
        all_centroid.append(new_centro_a)
        a=plt.show()
    
    lowest_cost_index = np.argmin(all_cost_function)
    
    cluster_label = all_closest_cluster[lowest_cost_index]
    cluster_centro = all_centroid[lowest_cost_index]
            
    u = np.unique(cluster_label)

    sns.set_theme(style="darkgrid")
    plt.figure(figsize = [14,9])
    for i in u: 
        plt.scatter(data[cluster_label == i , 0] , data[cluster_label == i , 1] , label = i)
    plt.scatter(cluster_centro[0][0],cluster_centro[0][1], marker = 'o', s=500, color = "b")
    plt.scatter(cluster_centro[1][0],cluster_centro[1][1], marker = 'o', s=500, color = "orange")
    plt.scatter(cluster_centro[2][0],cluster_centro[2][1], marker = 'o', s=500, color = "g")
    plt.scatter(cluster_centro[3][0],cluster_centro[3][1], marker = 'o', s=500, color = "r")
 
    plt.title("Kmean Clustering")
    plt.legend()
    plt.show()
            
    return cluster_label , cluster_centro, a

#Application using the simulation data in question 1

df = []
all_cluster_label=[]    #store each cluster label for different value of sigma
all_cluster_centroid = []  # store each cluster centroid for different value of sigma
for i in range(len(s)):
    
    data = np.zeros((100,2))
    data[0:25,0], data[0:25,1] = cluster1[i][0],  cluster1[i][1]
    data[25:50,0], data[25:50,1] = cluster2[i][0],  cluster2[i][1]
    data[50:75,0], data[50:75,1] = cluster3[i][0],  cluster3[i][1]
    data[75:100,0], data[75:100,1] = cluster4[i][0],  cluster4[i][1]  
    
    cluster_label_9 , cluster_centro_9, _ = Kmeans(data, K=4, valu_max=10000, tol=0.9)
    all_cluster_label.append(cluster_label_9)
    all_cluster_centroid.append(cluster_centro_9)

"""
To compute the quality of the obtained partition, we need to repeat the function step n times 
and keep the partition with the minimum within-cluster inertia (or sum of squares, or variance)

To compute the misclassification we need to take avec prediction(cluster label) and the class of each 
observation and make the confusion matrix to the error.
"""

#%%

"""
10. Generalize your function for a matrix X ∈ Rn×p , where p > 2.

"""

def Kmeans(data, K, n_iter, tol, n_repeated, random_state = 123):
    """
    Kmean clustering

    Parameters
    ----------
    data : TYPE : Array
        DESCRIPTION : Dataset
    K : TYPE : integer
        DESCRIPTION : Nulber of cluster
    n_iter : TYPE : integer
        DESCRIPTION : Number of iterations 
    tol : TYPE : float
        DESCRIPTION : Epsilon: value that the condition has to fulfill
    n_repeated : TYPE : integer
        DESCRIPTION : number of repeatition
    random_state : TYPE, optional
        DESCRIPTION. The default is 123.

    Returns
    -------
    cluster_label : TYPE : integer
        DESCRIPTION : Cluster label
    cluster_centro : TYPE : array
        DESCRIPTION : Cluster centroid
    
    Plot : Final plot of clustering

    """
    
    np.random.seed(random_state)
    n,k = np.array(data).shape
    
    all_cost_function=[]
    all_closest_cluster = []
    all_centroid = []

    for e in range(n_repeated):
        
        for it in range(n_iter):
            cent_1 = init_cent(data, K)
            _, min_c_1 = clust_assignment(data, cent_1, K)
        
            new_centro=mean_centr(data, min_c_1, K)
            new_centro_def = pd.DataFrame(new_centro)
            new_centro_a = np.array(new_centro_def)
            
            #K=new_centro.shape[0]
            if it:
                norm = np.linalg.norm(next_centroid - new_centro_a)
                if norm <tol:
                    print(it+1)
                    break
            next_centroid = new_centro_a
            
        J = np.zeros((n,k))
        cost_function = np.zeros((n,1))             
        for i in range(n):
                    
            for j in range(K):
                        
                if min_c_1[i]==j:
                            
                    J[i,:] = new_centro_a[j]
            cost_function[i] = (data[i][0]- J[i][0])**2 + (data[i][1]- J[i][1])**2
        all_cost_function.append(np.mean(cost_function)) 
        all_closest_cluster.append(min_c_1)
        all_centroid.append(new_centro_a)
        
       
    lowest_cost_index = np.argmin(all_cost_function)
    
    cluster_label = all_closest_cluster[lowest_cost_index]
    cluster_centro = all_centroid[lowest_cost_index]
    
    u = np.unique(cluster_label)

    sns.set_theme(style="darkgrid")
    plt.figure(figsize = [14,9])
    for i in u: 
        plt.scatter(data[cluster_label == i , 0] , data[cluster_label == i , 1] , label = i)
        plt.scatter(cluster_centro[i][0],cluster_centro[i][1], marker = 'o', s=500)
 
    plt.title("Kmean Clustering")
    plt.legend()
    plt.show()
            
    
            
    return cluster_label , cluster_centro


cluster_label_10 , cluster_centro_10 = Kmeans(data, K=4, n_iter=10000, tol=0.9, n_repeated = 100)

#%%

"""
Test
"""

from sklearn.cluster import KMeans

kmean = KMeans(n_clusters=4, random_state=123)

kmean.fit(data)
cluster_centro = kmean.cluster_centers_
sns.set_theme(style="darkgrid")
plt.figure(figsize = [14,9])

for i in np.unique(kmean.labels_): 
    plt.scatter(data[kmean.labels_ == i , 0] , data[kmean.labels_ == i , 1] , label = i)

plt.scatter(cluster_centro[0][0],cluster_centro[0][1], marker = 'o', s=500, color = "b")
plt.scatter(cluster_centro[1][0],cluster_centro[1][1], marker = 'o', s=500, color = "orange")
plt.scatter(cluster_centro[2][0],cluster_centro[2][1], marker = 'o', s=500, color = "g")
plt.scatter(cluster_centro[3][0],cluster_centro[3][1], marker = 'o', s=500, color = "r")
plt.show()

#%%

"""
11. Test your function on a real dataset.
"""

# import the module datasets of scikit-learn
from sklearn.datasets import load_iris
# load the dataset and store it in `data`
df = load_iris()
# the data (without the target variable) can be accessed
print(df.data)
# for informtion, the target is accessed as follows
print(df.target)


clus_lab_iris, clus_centro_iris = Kmeans(data = df.data, K = 3, n_iter=1000, tol = 0.1, n_repeated = 10)

"""
To compute the quality of the obtained partition, we need to repeat the function step n times 
and keep the partition with the minimum within-cluster inertia (or sum of squares, or variance)

To compute the misclassification we need to take avec prediction(cluster label) and the class of each 
observation and make the confusion matrix to the error.
"""

#%%

"""
References : 

    - Lectures
    - https://stackoverflow.com/questions/26007306/how-to-properly-stop-iteration-in-minimization-in-python
    - https://towardsdatascience.com/develop-your-own-newton-raphson-algorithm-in-python-a20a5b68c7dd
    - https://analyticsarora.com/k-means-for-beginners-how-to-build-from-scratch-in-python/
    
We have been inspired by these sources to write our own codes.

"""

  
