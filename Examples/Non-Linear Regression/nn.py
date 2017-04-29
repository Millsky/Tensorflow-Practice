import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Non-Linear Regression - via neural networks

# Params - Tensorflow Setup 
logs_path = "./logs/"

# Params - NN
learning_rate_s = .02
training_epochs = 250000

# training data represents the function (x^2) 

xt = np.random.random((2000,1)) * 10
y_t = np.power(xt,2)

# test data 
x_test  = np.array([  [1],[10],  [7], [11],[9],[23],[-1],[0],[-10],[-2]],dtype=np.float32)
y_test =    np.array([[1],[100],[49],[121],  [81],[529],[1],[0],[100],[4]],dtype=np.float32)

# tf Graph Input
X = tf.placeholder("float",[None,1],name="Placeholder-X")
Y = tf.placeholder("float",[None,1],name="Placeholder-Y")

# Set model weights
W = {
    'h1': tf.Variable(tf.random_uniform([1,140],dtype= tf.float32), name="weight-hidden-1"),
    'h2': tf.Variable(tf.random_uniform([140,1],dtype= tf.float32), name="weight-hidden-2")
}

# Set model biases 
B = {
    'b1': tf.Variable(tf.random_normal([1,140]), name="bias-hidden-1"),
    'b2': tf.Variable(tf.random_normal([1]), name="bias-hidden-2")
}

# Define Forward Pass Function 
def passForward(_X,_weights,_biases):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_X,_weights['h1']),_biases['b1']))
    return tf.nn.relu(tf.add(tf.matmul(layer_1,_weights['h2']),_biases['b2']))

# Prediction 
pred = passForward(X,W,B)

# Error 
error = tf.nn.l2_loss(passForward(X,W,B) - Y)

# Define optimization function 
train_step = tf.train.AdagradOptimizer(learning_rate_s).minimize(error)

# Define Extra Log Information for the tensorboard - we can add more later w/ merge v.1.0 
tf.summary.scalar("cost", error)
summary_op = tf.summary.merge_all()

# Init 
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
     # create log writer object
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    sess.run(init)
    for epoch in range(training_epochs):
        # Write to summary for tensorboard 
        _, summary = sess.run([train_step,summary_op], feed_dict={X: xt, Y: y_t})
        writer.add_summary(summary, epoch)