import tensorflow as tf
import numpy as np

# Linear Regression

# Params 
learning_rate = 0.001
training_epochs = 10000
display_step = 50

# Params - Tensorflow Setup 
# Tensorboard can be run with "tensorboard --logdir=path/to/log-directory" -> tensorboard --logdir=./logs/
logs_path = "./logs/"

# training data represents the function x+1

# training data x
train_X = np.array([1,2,3,4,5,6,7,8,9,10])

# training data y
train_Y = np.array([2,3,4,5,6,7,8,9,10,11])

n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(np.random.rand(), name="weight")
b = tf.Variable(np.random.rand(), name="bias")

# Prediction: we just do one pass [x,y] -(nn)> 0 single perceptron
pred = tf.add(tf.multiply(X, W), b)

# Error 
error = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

# Define Extra Log Information for the tensorboard - we can add more later w/ merge v.1.0 
tf.summary.scalar("cost", error)
summary_op = tf.summary.merge_all()

# Minimize Error 
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)

# Init 
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    # create log writer object
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    sess.run(init)
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            _, summary = sess.run([optimizer,summary_op], feed_dict={X: x, Y: y})
            writer.add_summary(summary, epoch)