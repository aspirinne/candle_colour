
# coding: utf-8

# In[3]:


# Based on Tensorflow example notebook:
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/multilayer_perceptron.ipynb


# In[4]:


import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[5]:


DATA_FILENAME = 'USDCB_170524_180524.csv'


# In[6]:


df = pd.read_csv(DATA_FILENAME)


# In[7]:


df.head()


# In[8]:


df.columns = map(lambda s: s[1:-1].lower(), df.columns)


# In[9]:


df.head()


# In[10]:


df['bar'] = (df['open'] >= df['close']).astype(int)


# In[11]:


df.head()


# In[12]:


df = df[['open', 'close', 'bar']]
X = df[['open', 'close']]
y = df['bar']
y = np.array([y, -(y-1)]).T


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)


# In[14]:


from sklearn.preprocessing import StandardScaler


# In[15]:


scaler = StandardScaler()


# In[16]:


X_train_scaled = scaler.fit_transform(X_train)


# In[17]:


X_test_scaled = scaler.transform(X_test)


# In[48]:


# Parameters
learning_rate = 0.001
training_epochs = 250
batch_size = 5
display_step = 1


# Network Parameters
n_hidden_1 = 2 # 1st layer number of features
n_hidden_2 = 2 # 2nd layer number of features
n_input = 2 # Number of feature
n_classes = 2 # Number of classes to predict


# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    #drop_out = tf.nn.dropout(layer_2, 0.01)
    #out_layer = tf.matmul(drop_out, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()


# In[49]:


# Launch the graph

with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(X_train_scaled)/batch_size)
        X_batches = np.array_split(X_train_scaled, total_batch)
        Y_batches = np.array_split(y_train, total_batch)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: X_test_scaled, y: y_test}))
    global result 
    result = tf.argmax(pred, 1).eval({x: X_test_scaled, y: y_test})

