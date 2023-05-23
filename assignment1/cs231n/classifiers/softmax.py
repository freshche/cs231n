from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    C = W.shape[1]

    data_loss = 0
    for i in range(N):
      log_counts = W.T @ X[i].T
      counts = np.exp(log_counts)
      sum_counts = np.sum(counts)

      for j in range(C):
        count_j_class = counts[j]
        prob_j_class =  count_j_class / sum_counts
        if j == y[i]:
          dW[:, j] += (X[i] * (prob_j_class - 1))
          data_loss += -np.log(prob_j_class)
        else:
          dW[:, j] += (X[i] * prob_j_class)

    reg_loss = reg * np.sum(W ** 2)
    loss = data_loss + reg_loss

    loss /= N
    dW /= N 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, D = X.shape[0], X.shape[1]
    log_counts = W.T @ X.T
    counts = np.exp(log_counts)
    sum_counts = np.sum(counts, axis=0)
    probs_classes = counts / sum_counts
    
    probs_gt_classes = probs_classes[y, range(N)]
    data_losses = -np.log(probs_gt_classes)
    data_loss = np.sum(data_losses) / N

    reg_loss = reg * np.sum(W ** 2) 
    loss = data_loss + reg_loss

    probs_classes[y, np.arange(0, probs_classes.shape[1])] -= 1
    dW_T = probs_classes @ X
    dW = dW_T.T / N
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
