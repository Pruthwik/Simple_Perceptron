#!/usr/bin/env python
# coding: utf-8

# In[1]:


from random import random


# In[2]:


random()


# In[3]:


input_shape = 3


# In[4]:


inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]


# In[5]:


def augment_inputs(inputs):
    return [(1, x[0], x[1]) for x in inputs]


# In[6]:


augment_inputs(inputs)


# In[7]:


def initialize_weights(input_shape):
    return [random() for i in range(input_shape)]


# In[28]:


W = initialize_weights(input_shape)


# In[9]:


W


# In[29]:


classes = [0, 0, 0, 1]
import numpy as np
W = np.array(W)


# In[30]:


def simple_perceptron(X, W, classes, itr=1):
    print(X, W)
    for i in range(itr):
        sat = list()
        for ind, x in enumerate(X):
            x = np.array(x)
            if W.dot(x) <= 0 and classes[ind] == 1:
                W += x
                sat.append(False)
            elif W.dot(x) > 0 and classes[ind] == 0:
                W -= x
                sat.append(False)
            else:
                sat.append(True)
            print(W.dot(x), classes[ind], ind)
        if np.all(sat):
            print('i=', i)
            return W
    return W


# In[31]:


augmented_inputs = augment_inputs(inputs)
final_W = simple_perceptron(augmented_inputs, W, classes, 10)


# In[21]:


final_W


# In[ ]:




