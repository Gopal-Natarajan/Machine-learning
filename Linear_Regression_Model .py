#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax


# ### Loading Data

# In[2]:


data = pd.read_csv("data_for_lr.csv")


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.shape


# In[6]:


data.isnull().sum()


# #### y colunm having one null value. i am going to drop that null value

# In[7]:


data = data.dropna()


# In[8]:


data.shape


# # Splitting up the data for test and train our model

# In[40]:


#training input and output 
training_input = np.array(data.x[0:500]).reshape(500,1)
training_output = np.array(data.y[0:500]).reshape(500,1)

#testing input and output
test_input = np.array(data.x[500:700]).reshape(199,1)
test_output = np.array(data.y[500:700]).reshape(199,1)


# In[10]:


np.array(data.x[0:500])


# In[11]:


np.array(data.x[0:500]).reshape(500,1)


# In[12]:


# Shape of the training ang testing input and output data

print(f'training input data shape = {training_input.shape}')
print(f'training output data shape = {training_output.shape}')
print(f'testing input data shape = {test_input.shape}')
print(f'testing output data shape = {test_output.shape}')


# # <font color = "green">Linear Regration</font>

# ## Forward propagation

# f(x) = m*x + c
# 
# At the time of forwar propagation we know our input data and output data. We need to know the parameters.\
# So, in the forward propagation function you need to pass input data and parameters
# 
# X = input data \
# M,C = parameters

# In[13]:


def forward_propagation(training_input, parameters):
    m = parameters['m']
    c = parameters['c']

    predictions = np.multiply(m,training_input) + c

    return predictions


# ## Cost Function

# cost = summation(1/2n * (y - f(x))^2)
# 
# in the cost funtion n,y,f(x) is the most important once.\
# In the cost function we should pass the prediction and training_output
# 
# Y = output data \
# f(x) = predictions
# 

# In[14]:


def cost_function(predictions, training_output):

    cost = np.mean((training_output - predictions) ** 2) * 0.5

    return cost


# ## Gradient Descent for Backpropagation

# df = summation(f(x) - y) \
# dm = df * x \
# dc = df * 1

# In[ ]:





# In[15]:


def backward_propagation(training_input, training_output, predictions):

    derivatives = dict()

    df = predictions - training_output
    dm = np.mean(np.multiply(df , training_input))
    dc = np.mean(df)

    derivatives['dm'] = dm
    derivatives["dc"] = dc

    return derivatives


# ### Update parameters
# 
# m = m - (learning_rate * dm) \
# 
# c = c - (learning_rate * dc)

# In[16]:


def update_parameters(parameters, derivatives, learning_rate):

    parameters['m'] = parameters['m'] - learning_rate * derivatives['dm']
    parameters['c'] = parameters['c'] - learning_rate * derivatives['dc']

    return parameters
    


# # Model Training

# In[17]:


def train(training_input, training_output, learning_rate, iters):

    #Random parameters

    parameters = dict()
    parameters['m'] = np.random.uniform(0,1)
    parameters['c'] = np.random.uniform(0,1)

    plt.figure()

    #loss
    loss = list()
    
    #iteration
    for i in range(iters):

        #Forward propagation
        predictions = forward_propagation(training_input, parameters)

        #cost
        cost = cost_function(predictions, training_output)

        loss.append(cost)
        print(f'Iteration = {i+1} , Loss = {cost}')

        #plot
        fig, ax = plt.subplots()

        ax.plot(training_input, training_output, '+', label = "Original")
        ax.plot(training_input, predictions, "*", label= "Training")

        legend = ax.legend()


        ax.plot(training_input, training_output, '+', label = "Original")
        ax.plot(training_input, predictions, "*", label= "Training")
        plt.show()
        #backward function
        derivatives = backward_propagation(training_input, training_output, predictions)

        #update the parameters
        parameters = update_parameters(parameters, derivatives, learning_rate)

    return parameters, loss
    


# # Traning

# In[18]:


parameters, loss = train(training_input,training_output, 0.0001, 20)


# In[19]:


print(parameters)


# In[20]:


loss


# In[21]:


plt.plot(loss)


# # Let's predict

# In[22]:


test_prediction = test_input * parameters['m'] + parameters['c']


# In[23]:


plt.plot(test_input, test_prediction, '+')
plt.plot(test_output, test_prediction, '*')
plt.show()


# # cost of prediction

# In[24]:


cost_function(test_output, test_prediction)


# # Linear Regression using Sk-learn

# In[27]:


from sklearn.linear_model import LinearRegression


# In[28]:


lr_model = LinearRegression()


# In[30]:


lr_model.fit(training_input, training_output)


# In[31]:


lr_model.coef_


# In[32]:


lr_model.intercept_


# # Prediction

# In[33]:


test_predictions = lr_model.predict(test_input)


# In[35]:


plt.plot(test_input, test_predictions, '+')
plt.plot(test_output, test_predictions, '*')
plt.xlabel('input')
plt.ylabel('Output/predictions')
plt.title('Performance Testing')
plt.show()


# # Cost

# In[37]:


from sklearn.metrics import mean_squared_error

cost = mean_squared_error(test_output, test_predictions)
cost


# In[ ]:




