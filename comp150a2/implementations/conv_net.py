"""
Implementation of convolutional neural network. Please implement your own convolutional neural network 
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class ConvNet(object):
  """
  A convolutional neural network. 
  """

  def __init__(self, input_size, output_size, filter_size, pooling_schedule, fc_hidden_size,  weight_scale=None, centering_data=False, use_dropout=False, use_bn=False):
    """
    A suggested interface. You can choose to use a different interface and make changes to the notebook.

    Model initialization.

    Inputs:
    - input_size: The dimension D of the input data.
    - output_size: The number of classes C.
    - filter_size: sizes of convolutional filters
    - pooling_schedule: positions of pooling layers 
    - fc_hidden_size: sizes of hidden layers of hidden layers 
    - weight_scale: the initialization scale of weights
    - centering_data: Whether centering the data or not
    - use_dropout: whether use dropout layers. Dropout rates will be specified in training
    - use_bn: whether to use batch normalization

    Return: 
    """
    tf.reset_default_graph()

    # record all options
    self.options = {'centering_data':centering_data, 'use_dropout':use_dropout, 'use_bn':use_bn}


    self.num_layers = len(filter_size) + len(fc_hidden_size) + 1
    hidden_size = [input_size] + np.shape(filter_size)[0]*[input_size]+fc_hidden_size+[output_size]

    #print(np.shape(filter_size))
    #print(hidden_size)

    # adjust layer sizes to account for filter reshaping
    n_filter_layers = np.shape(filter_size)[0]
    for ii in range(n_filter_layers):
        hidden_size[ii+1] = [input_size[0],input_size[1],filter_size[ii][-1]]
        for jj in pooling_schedule:
            if ii+1 > jj:
                hidden_size[ii+1][0:2] = [int(x/2) for x in hidden_size[ii+1][0:2]]

    #print('Hidden Size: %s' %hidden_size)

    # get filter parameters
    filter_reshaped = np.reshape(filter_size,np.shape(filter_size)[:])
    kernel_sizes = filter_reshaped[:,:-1]
    filters = filter_reshaped[:,-1]

    # construct the computational graph 
    #self.tf_graph = tf.Graph()
    #with self.tf_graph.as_default():

    # allocate parameters
    self.params = {'W': [], 'b': [], 'filter': []}

    # FC weights

    filter_counter = 0
    for ilayer in range(self.num_layers): 
        # the scale of the initialization

        if weight_scale is None:
            weight_scale = np.sqrt(2 / np.product(hidden_size[ilayer]))

        if ilayer == (self.num_layers - 1) or ilayer == (self.num_layers-2):
            W = tf.Variable(weight_scale * np.random.randn(np.product(hidden_size[ilayer]), hidden_size[ilayer + 1]), dtype=tf.float32)
            b = tf.Variable(0.01 * np.ones(hidden_size[ilayer + 1]), dtype=tf.float32)
            self.params['W'].append(W)
            self.params['b'].append(b)

        else:
            filter_height,filter_width,out_channels = filter_size[filter_counter][0],filter_size[filter_counter][1],hidden_size[ilayer+1][2]
            in_channels = hidden_size[ilayer][2]
            filter_shape = [filter_height,filter_width,in_channels,out_channels]

            current_filter = tf.Variable(np.random.normal(size = filter_shape, scale = 10*weight_scale), dtype=tf.float32)
            self.params['filter'].append(current_filter)
            filter_counter += 1


    # allocate convolutional parameters
    self.conv_params = {}

    # filter parameters
    self.conv_params['kernel_sizes'] = kernel_sizes
    self.conv_params['filters'] = filters

    # pooling schedule
    self.conv_params['pooling_schedule'] = pooling_schedule

    # allocate place holders 
    self.placeholders = {}

    # data feeder
    self.placeholders['x_batch'] = tf.placeholder(dtype=tf.float32, shape=np.hstack([None,input_size]))
    self.placeholders['y_batch']= tf.placeholder(dtype=tf.int32, shape=[None])

    # the working mode 
    self.placeholders['training_mode'] = tf.placeholder(dtype=tf.bool, shape=())
    
    # data center 
    self.placeholders['x_center'] = tf.placeholder(dtype=tf.float32, shape=input_size)

    # keeping probability of the dropout layer
    self.placeholders['keep_prob'] = tf.placeholder(dtype=tf.float32, shape=[])

    # regularization weight 
    self.placeholders['reg_weight'] = tf.placeholder(dtype=tf.float32, shape=[])

    # learning rate
    self.placeholders['learning_rate'] = tf.placeholder(dtype=tf.float32, shape=[])
    
    self.operations = {}

    # construct graph for score calculation 
    scores = self.compute_scores(self.placeholders['x_batch'])
                            
    # predict operation
    self.operations['y_pred'] = tf.argmax(scores, axis=-1)

    # construct graph for training 
    objective = self.compute_objective(scores, self.placeholders['y_batch'])
    self.operations['objective'] = objective

    minimizer = tf.train.GradientDescentOptimizer(learning_rate=self.placeholders['learning_rate'])
    training_step = minimizer.minimize(objective)

    self.operations['training_step'] = training_step 

    if self.options['use_bn']:
      bn_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    else: 
      bn_update = []
    self.operations['bn_update'] = bn_update

    # maintain a session for the entire model
    self.session = tf.Session()

    self.x_center = None # will get data center at training

    return 

  def softmax_loss(self, scores, y):
    """
    Compute the softmax loss. Implement this function in tensorflow

    Inputs:
    - scores: Input data of shape (N, C), tf tensor. Each scores[i] is a vector 
              containing the scores of instance i for C classes .
    - y: Vector of training labels, tf tensor. y[i] is the label for X[i], and each y[i] is
         an integer in the range 0 <= y[i] < C. This parameter is optional; if it
         is not passed then we only return scores, and if it is passed then we
         instead return the loss and gradients.
    - reg: Regularization strength, scalar.

    Returns:
    - loss: softmax loss for this batch of training samples.
    """
    softmax_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=scores))

    return softmax_loss

  def regularizer(self):
    """ 
    Calculate the regularization term
    Input: 
    Return: 
        the regularization term
    """
    reg = np.float32(0.0)
    for W in self.params['W']:
        reg = reg + self.placeholders['reg_weight'] * tf.reduce_sum(tf.square(W))
    
    return reg

  def compute_scores(self, X):
    """

    Compute the loss and gradients for a two layer fully connected neural
    network. Implement this function in tensorflow

    Inputs:
    - X: Input data of shape (N, D), tf tensor. Each X[i] is a training sample.

    Returns:
    - scores: a tensor of shape (N, C) where scores[i, c] is the score for 
              class c on input X[i].

    """
    # Unpack variables from the params dictionary
    if self.options['centering_data']:  
        X = X - self.placeholders['x_center']

    #num_layers = len(self.params['W'])

    hidden = X

    W_counter = 0
    filter_counter = 0
    for ilayer in range(0, self.num_layers): 
        #print(ilayer)
        #W = self.params['W'][ilayer]
        #b = self.params['b'][ilayer]
        #W_filter = self.params['filter'][ilayer]
        
        #print('hidden shape: %s' %hidden.get_shape())
        #print('W: %s' %W.get_shape())
        #linear_trans = tf.matmul(hidden, W) + b

        # if the last layer, then the linear transformation is the end
        if ilayer > (self.num_layers - 3):

            W = self.params['W'][W_counter]
            b = self.params['b'][W_counter]
            #print('Hidden Shape: %s' %hidden.get_shape())
            #print('W Shape: %s' %W.get_shape())

            hidden = tf.layers.flatten(hidden)
            linear_trans = tf.matmul(hidden,W) + b
            hidden = linear_trans

            W_counter += 1

        # otherwise optionally apply batch normalization, relu, and dropout to all layers 
        else:

            # convolutional layer
            #conv = tf.layers.conv2d(hidden,self.conv_params['filters'][ilayer],self.conv_params['kernel_sizes'][ilayer], padding = 'same')
            W_filter = self.params['filter'][filter_counter]

            conv = tf.nn.conv2d(hidden,self.params['filter'][filter_counter],[1,1,1,1] ,'SAME')
            
            # batch normalization
            if self.options['use_bn']:
              batched = tf.layers.batch_normalization(conv, training = self.placeholders['training_mode'], momentum = 0.95)
            else:
              batched = conv

            # non-linear transformation
            relu = tf.nn.relu(batched)

            # dropout
            if self.placeholders['training_mode'] == True:
              dropped = tf.nn.dropout(relu, keep_prob = self.placeholders['keep_prob'])
            else:
              dropped = relu

            # pooling
            if ilayer in self.conv_params['pooling_schedule']:
                hidden = tf.layers.max_pooling2d(dropped,[2,2],2)
            else: 
                hidden = dropped

            filter_counter += 1
        
    scores = hidden

    return scores

  def compute_objective(self, scores, y):
    """
    Compute the training objective of the neural network.

    Inputs:
    - scores: A numpy array of shape (N, C). C scores for each instance. C is the number of classes 
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - reg: a np.float32 scalar

    Returns: 
    - objective: a tensorflow scalar. the training objective, which is the sum of 
                 losses and the regularization term
    """

    # get output size, which is the number of classes
    num_classes = self.params['b'][-1].get_shape()[0]

    y1hot = tf.one_hot(y, depth=num_classes)
    loss = self.softmax_loss(scores, y1hot)

    reg_term = self.regularizer()

    objective = loss + reg_term

    return objective

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=1.0, keep_prob=1.0, 
            reg=np.float32(5e-6), num_iters=100,
            batch_size=200, verbose=False):
    """
    A suggested interface of training the model.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - keep_prob: probability of keeping values when using dropout
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)
    num_classes = self.params['b'][-1].get_shape()[0]
    #num_layers = len(self.params['W'])

    self.x_center = np.mean(X, axis=0)
    self.params_loaded = {'W': [], 'b': [], 'filter': []}

    ############################################################################
    # after this line, you should execute appropriate operations in the graph to train the mode  

    session = self.session
    session.run(tf.global_variables_initializer())

    # Use SGD to optimize the parameters in self.model
    objective_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):

      b0 = (it * batch_size) % num_train 
      batch = range(b0, min(b0 + batch_size, num_train))

      X_batch = X[batch]
      y_batch = y[batch] 

      feed_dict = {self.placeholders['x_batch']: X_batch, 
                   self.placeholders['y_batch']: y_batch, 
                   self.placeholders['learning_rate']:learning_rate, 
                   self.placeholders['training_mode']:True, 
                   self.placeholders['reg_weight']:reg}

      # Decay learning rate
      learning_rate *= learning_rate_decay


      if self.options['centering_data']:
        feed_dict[self.placeholders['x_center']] = self.x_center

      if self.options['use_dropout']:
        feed_dict[self.placeholders['keep_prob']] = np.float32(keep_prob)
     
     
      np_objective, _, _  = session.run([self.operations['objective'], 
                                     self.operations['training_step'],
                                     self.operations['bn_update']], feed_dict=feed_dict)

      objective_history.append(np_objective) 

      if verbose and it % 100 == 0:
        print('iteration %d / %d: objective %f' % (it, num_iters, np_objective))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = np.float32(self.predict(X_batch) == y_batch).mean()
        val_acc = np.float32(self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

    self.params_loaded['W'] = session.run(self.params['W'])
    self.params_loaded['b'] = session.run(self.params['b'])
    self.params_loaded['filter'] = session.run(self.params['filter'])

    return {
      'objective_history': objective_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of the neural network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """

    np_y_pred = self.session.run(self.operations['y_pred'], feed_dict={self.placeholders['x_batch']: X, 
                                                                       self.placeholders['training_mode']:False, 
                                                                       self.placeholders['x_center']:self.x_center, 
                                                                       self.placeholders['keep_prob']:1.0} 
                                                                       )

    return np_y_pred

  def get_params(self):
    """
    Returns parameters of the network
    """
    return(self.params_loaded)


