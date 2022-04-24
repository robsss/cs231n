from builtins import range
from turtle import circle
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
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores = X.dot(W)
    logc = 0.0
    for i in range(num_train): 
      denom = 0.0
      p_row = np.zeros(num_classes)
      # numerical stability factor
      logc = -1 * scores[i].max()

      # denominator
      for j in range(num_classes):
        denom += np.exp(scores[i][j] + logc)

      # gradient 
      for j in range(num_classes):
        if j == y[i]:
          dW[:,y[i]] += -X[i] + X[i]*np.exp(scores[i][y[i]] + logc) / denom
          continue
        dW[:,j] += X[i] * np.exp(scores[i][j] + logc) / denom

      loss += -1 * np.log(np.exp(scores[i][y[i]] + logc) / (denom))
    
    # Average 
    dW /= num_train
    loss /= num_train

    # regularization
    dW += reg * W
    loss += 0.5 * reg * np.sum(W**2)
    pass

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
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # scores
    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores = X.dot(W)

    # numerical stability
    logc = -1 * scores.max(axis=1).reshape(num_train,1)
    exp_scores = np.exp(scores + logc)
    correct_scores = exp_scores[range(num_train),y]
    denom = np.sum(exp_scores,axis=1)
    loss_row = -1 * np.log(correct_scores / denom)

    # gradient
    p = exp_scores / denom.reshape((num_train,1))
    negative_term = np.zeros((num_train, num_classes))
    negative_term[range(num_train), y] = -1
    dW = X.T.dot(p + negative_term)

    # average
    loss = np.sum(loss_row) / num_train
    dW /= num_train
  
    # regularization
    dW += reg * W
    loss += 0.5 * reg * np.sum(W**2)
    pass
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
