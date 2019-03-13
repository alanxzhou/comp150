import numpy as np
import tensorflow as tf
import warnings

def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.

    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    """
    mode = bn_param['mode']
    eps = bn_param['eps']
    momentum = bn_param['momentum']
    running_mean = bn_param['running_mean']
    running_var = bn_param['running_var'] 

    out = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out.                    #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        mu = x.mean(axis = 0)
        var = x.var(axis = 0)
        x = (x-mu)/np.sqrt(var)
        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * var
        #x = (x-running_mean)/np.sqrt(running_var)
        x = gamma * x + beta

        out = x
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        x = x - running_mean.astype(float)
        x = x / np.sqrt(running_var)
        x = gamma * x + beta
        out = x
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
    return out




def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    Namely, divide the the training output by p, and do nothing for testing

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        sess = tf.Session()
        dropped = tf.nn.dropout(x, keep_prob = p)
        out = sess.run(dropped)
        #out = tf.Session().run(tf.nn.dropout(x, keep_prob=p))
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    return out



def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, H, W, C)
    - w: Filter weights of shape (HH, WW, C, F)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, H', W', F) where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    """
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################

    stride = conv_param['stride']
    pad = conv_param['pad']

    # padding
    npad = ((0,0),(pad,pad),(pad,pad),(0,0))
    x_padded = np.pad(x, npad, 'constant', constant_values = 0)

    [N, H, W, C] = np.shape(x)
    [HH, WW, _, F] = np.shape(w)
    H_new = int(1 + (H + 2 * pad - HH) / stride)
    W_new = int(1 + (W + 2 * pad - WW) / stride)
    out = np.zeros((N, H_new, W_new, F))

    for nn in range(N):
    	for hh in range(0,H_new):
    		for ww in range(0,W_new):
    			for ff in range(F):
    				out[nn,hh,ww,ff] = np.sum(x_padded[nn, hh*stride:hh*stride+HH, ww*stride:ww*stride+WW,:]*w[:,:,:,ff])+b[ff]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out



def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, H, W, C)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    """
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################

    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    [N, H, W, C] = np.shape(x)
    HH = int(1 + (H - pool_height) / stride)
    WW = int(1 + (W - pool_width) / stride)
    out = np.zeros((N,HH,WW,C))

    for nn in range(N):
    	for hh in range(HH):
    		for ww in range(WW):
		    	for cc in range(C):

	    			#print(x[nn,hh*stride:hh*stride+HH,ww*stride:ww*stride+WW,cc])
    				out[nn,hh,ww,ccw] = np.max(x[nn,hh*stride:hh*stride+HH,ww*stride:ww*stride+WW,cc])

	#out = 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


    return out


