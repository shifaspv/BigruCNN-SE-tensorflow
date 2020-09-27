import numpy as np
import scipy as sp
import tensorflow as tf


def conv2D1(X, W,P):
#    if type(X) is list:
#        Xprev, Xcur, Xnext = X
#
#        Wprev = W[0, 0, :, :]
#        Wcur = W[0, 1, :, :]
#        Wnext = W[0, 2, :, :]
#        
#        conv_out = tf.matmul(Xprev, Wprev) + tf.matmul(Xcur, Wcur) + tf.matmul(Xnext, Wnext) 
#        out_shape = tf.shape(conv_out) 
#        conv_out = tf.reshape(conv_out, (1, 1, 1, out_shape[1]))
#    else:   
    conv_out = tf.nn.convolution(X, W, strides=[1,1,1,1], padding='SAME') 
    conv_out=tf.expand_dims(conv_out[:,:,P,:], axis=-2)

    return conv_out

def conv2D2(X, W):
#    if type(X) is list:
#        Xprev, Xcur, Xnext = X
#
#        Wprev = W[0, 0, :, :]
#        Wcur = W[0, 1, :, :]
#        Wnext = W[0, 2, :, :]
#        
#        conv_out = tf.matmul(Xprev, Wprev) + tf.matmul(Xcur, Wcur) + tf.matmul(Xnext, Wnext) 
#        out_shape = tf.shape(conv_out) 
#        conv_out = tf.reshape(conv_out, (1, 1, 1, out_shape[1]))
#    else:   
    conv_out = tf.nn.convolution(X, W, strides=[1,1,1,1], padding='SAME') 

    return conv_out
def conv2D3(X, W, strides):
#    if type(X) is list:
#        Xprev, Xcur, Xnext = X
#
#        Wprev = W[0, 0, :, :]
#        Wcur = W[0, 1, :, :]
#        Wnext = W[0, 2, :, :]
#        
#        conv_out = tf.matmul(Xprev, Wprev) + tf.matmul(Xcur, Wcur) + tf.matmul(Xnext, Wnext) 
#        out_shape = tf.shape(conv_out) 
#        conv_out = tf.reshape(conv_out, (1, 1, 1, out_shape[1]))
#    else:   
    conv_out = tf.nn.convolution(X, W, strides=[1,strides[0],strides[1],1], padding='VALID') 

    return conv_out




def int_shape(x):
    return list(map(int, x.get_shape()))

def concat_relu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    axis = 3
    return tf.nn.relu(tf.concat([x, -x], axis=axis))


