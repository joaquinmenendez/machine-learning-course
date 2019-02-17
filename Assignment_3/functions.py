import numpy as np
import pandas as pd
def logistic_prob (dataset,weight):
    values, new_val = [], []
    for n , row in enumerate(dataset):
        values.append(row @ weight.T )
    print (values)
    for mxi in values:
        print(-mxi)
        new_val.append((1/(1+np.exp(-mxi)))[0])
    return new_val # np arrray of len = to dataset and weight len only one value

a = np.array([[1,1,3],
             [1,1,0]])
w =  np.array([[.6,.9,.2]])
logistic_prob(a,w)


def cost_function(dataset,weight, normalize = True):
    mxp = []
    for n , row in enumerate(dataset):
        mxp.append(weight[n].T @ row)
    sum_val = 0
    for n,row in enumerate(dataset):
        y = row[2]
        sum_val = sum_val + (mxp[n] * (1 - y) + np.log(1 + np.exp(-mxp[n])))
    if normalize:
        norm_sum_val = -1*sum_val/(dataset.shape[0])
        return norm_sum_val
    else: return -1*sum_val #Should I return two values? one per class?

learning_rate = 0.001 #eta   learning rate should be something close to 10-3 to 10-6
stopping_criteria = 10^-6
w = np.array([np.random.random(),np.random.random(),np.random.random()]) #Initialize randomly some weights
    
def grad_eq(x,y,w_old,learning_rate): #The normalization is going to be provided by the sigmoid func
    w_old-(learning_rate* sum())

def prob_single_data(dataset,w):
    x = dataset.drop(['y'], axis = 1)
    x['x0'] = np.array([1]*dataset.shape[0]) # will add ones to X0
    y = dataset['y']
    return sum(x*(1-y)*logistic_prob)



def batch_gradient_desc(dataset,weight, normalize = True):
    #fitting model for every w  
    .value
    
    
    
    
    
    
    
    
    
    
    
    
    