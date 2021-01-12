import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import re 
from collections import Counter
from utils import sigmoid, get_batches, compute_pca, get_dict 
import streamlit as st 




def initialize_model(N,V, random_seed=1):
    np.random.seed(random_seed)
    W1 = np.random.rand(N,V)
    W2 = np.random.rand(V,N)
    b1 = np.random.rand(N,1)
    b2 = np.random.rand(V,1)

    return W1, W2, b1, b2


def softmax(z):
    yhat = np.exp(z) /  np.sum(np.exp(z) , axis=0)
    return yhat

def forward_prop(x, W1, W2, b1, b2):
    h = np.dot(W1,x) +b1
    h = np.maximum(0,h)
    z = np.dot(W2,h) + b2
    return z, h

def compute_cost(y, yhat, batch_size):
    logprobs = np.multiply(np.log(yhat),y) + np.multiply(np.log(1 - yhat), 1 - y)
    cost = - 1/batch_size * np.sum(logprobs)
    cost = np.squeeze(cost)
    return cost

def back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size):
 
    l1 = np.dot(W2.T,yhat-y)
    # Apply relu to l1
    l1 = np.maximum(0,l1)
    # Compute the gradient of W1
    grad_W1 = (1/batch_size) *np.dot(l1,x.T)
    # Compute the gradient of W2
    grad_W2 = (1/batch_size) * np.dot(yhat-y,h.T)
    # Compute the gradient of b1
    grad_b1 = np.sum((1/batch_size)*l1,axis=1,keepdims = True) 
    # Compute the gradient of b2
    grad_b2 =np.sum((1/batch_size)*(yhat-y),axis=1,keepdims = True)
    ### END CODE HERE ###
    
    return grad_W1, grad_W2, grad_b1, grad_b2 



def gradient_descent(data, word2Ind, N, V, num_iters, alpha=0.03):
 
    W1, W2, b1, b2 = initialize_model(N,V, random_seed=282)
    batch_size = 128
    iters = 0
    C = 2
 
    for x, y in get_batches(data, word2Ind, V, C, batch_size):
 
        z, h = forward_prop(x, W1, W2, b1, b2)
      
        yhat = softmax(z)
        
        cost = compute_cost(y, yhat, batch_size)
        if ( (iters+1) % 10 == 0):
            st.write(f"iters: {iters + 1} cost: {cost:.6f}")
 
        grad_W1, grad_W2, grad_b1, grad_b2 = back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size)
        W1 = W1-alpha* grad_W1
        W2 = W2-alpha*grad_W2
        b1 = b1 - alpha * grad_b1
        b2 = b2 - alpha * grad_b2
        
        iters += 1 
        if iters == num_iters: 
            break
        if iters % 100 == 0:
            alpha *= 0.66
            
    return W1, W2, b1, b2 


