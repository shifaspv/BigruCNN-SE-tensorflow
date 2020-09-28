#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 09:57:15 2018

@author: Muhammed Shifas Pv
University of Crete (UoC)
"""
from __future__ import division
import os
import sys
import math
import logging
import time
import numpy as np
import lib.util as util
import tensorflow as tf
from datetime import datetime
from lib.ops import  conv2D1, conv2D2, conv2D3
from lib.util import l1_l2_loss
from lib.precision import _FLOATX
from lib.model_io import save_variables, get_info
from lib.util import compute_receptive_field_length
import pdb
from numpy.lib import stride_tricks
from numpy import inf
from lib.model_io import restore_variables
#from sklearn.metrics import r2_score

def get_var_maybe_avg(var_name, ema, **kwargs):
    ''' utility for retrieving polyak averaged params '''
    v = tf.get_variable(var_name, **kwargs)
    if ema is not None:
        v = ema.average(v)
    return v

def get_weight_variable(name, shape=None, initializer=tf.contrib.layers.xavier_initializer_conv2d(), ema=None):
    if shape is None:
        return get_var_maybe_avg(name, ema)
    else:  
        return get_var_maybe_avg(name, ema, shape=shape, dtype=_FLOATX, initializer=initializer)

def get_bias_variable(name, shape=None, initializer=tf.constant_initializer(value=0.0, dtype=_FLOATX), ema=None): 
    if shape is None:
        return get_var_maybe_avg(name, ema)
    else:  
        return get_var_maybe_avg(name, ema, shape=shape, dtype=_FLOATX, initializer=initializer)
   
def get_getter(ema):
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var
    return ema_getter




class BigruCNN_SE(object):
    
    def __init__(self,cfg,model_id=None):
        self.cfg=cfg        
        self.use_biases = cfg['use_biases']
        self.l2 = cfg['L2_regularization'] 
        self.use_dropout = cfg['use_dropout']
        self.use_ema = cfg['use_ema']
        self.polyak_decay = cfg['polyak_decay']
        self.num_gru_layers=cfg['num_gruCNN_layers']
        self.model_id = model_id    
        self.masker_length= cfg['fft_bin_size']/2
        self.filter_size_gruCNN=cfg[ 'filter_size_gruCNN']
        self.num_input_frames=cfg['num_input_frames']
        self.num_channel_gruCNN=cfg['num_channel_gruCNN']
        self.batch_size=cfg['batch_size']
        self.frame_size=int(cfg['frame_size']*cfg['sample_rate'])
        self.frame_shift=int(cfg['frame_shift']*cfg['sample_rate'])
        self.stride=cfg['gruCNN_filter_stride']
        self.create_variables()
        if self.use_ema:
            self.ema = tf.train.ExponentialMovingAverage(decay=self.polyak_decay)
        else:
            self.ema = None 
    
        
    def create_variables(self):       
        fw_x=self.filter_size_gruCNN["X_filter"]
        fw_h=self.filter_size_gruCNN["H_filter"]
        DFm=int((self.masker_length/4+1))
        r_gru=self.num_channel_gruCNN
        dim_out = self.masker_length+1
        with tf.variable_scope('BigruCNN_SE'):
                      
                
            with tf.variable_scope('BigruCNN_layer'):  
                with tf.variable_scope('Forward'):
                    
                    for i, feature_dim in enumerate(self.num_gru_layers):   
                        with tf.variable_scope('block{}'.format(i)):
                            fw_x_gru=fw_x["Layer{}".format(i+1)]
                            fw_h_gru=fw_h["Layer{}".format(i+1)]
    #            with tf.variable_scope('Conv_LSTM_layer'): # implementing LSTM manually
                            if i==0:
                                get_weight_variable('W_zx',( fw_x_gru[0], fw_x_gru[1], 1, r_gru))
                                get_weight_variable('W_zh',( fw_h_gru[0], fw_h_gru[1], r_gru, r_gru))
                                get_weight_variable('W_rx',(fw_x_gru[0], fw_x_gru[1], 1, r_gru))
                                get_weight_variable('W_rh',(fw_h_gru[0], fw_h_gru[1], r_gru, r_gru))
                                get_weight_variable('W_hx',(fw_x_gru[0],fw_x_gru[1], 1, r_gru))
                                get_weight_variable('W_hh',(fw_h_gru[0], fw_h_gru[1], r_gru, r_gru)) 
        #                        get_weight_variable('TD_W', (DFm*r_lstm,dim_out))  
                            else:
                                get_weight_variable('W_zx',( fw_x_gru[0],fw_x_gru[1],  r_gru, r_gru))
                                get_weight_variable('W_zh',( fw_h_gru[0],fw_h_gru[1] ,  r_gru, r_gru))
                                get_weight_variable('W_rx',(fw_x_gru[0], fw_x_gru[1], r_gru, r_gru))
                                get_weight_variable('W_rh',(fw_h_gru[0], fw_h_gru[1],  r_gru, r_gru))
                                get_weight_variable('W_hx',(fw_x_gru[0], fw_x_gru[1], r_gru, r_gru))
                                get_weight_variable('W_hh',(fw_h_gru[0], fw_h_gru[1],  r_gru, r_gru))    
                            get_weight_variable('alpha',(r_gru))       
                            
                with tf.variable_scope('Backward'):
                    
                    for i, feature_dim in enumerate(self.num_gru_layers):   
                        with tf.variable_scope('block{}'.format(i)):
                            fw_x_gru=fw_x["Layer{}".format(i+1)]
                            fw_h_gru=fw_h["Layer{}".format(i+1)]
    #            with tf.variable_scope('Conv_LSTM_layer'): # implementing LSTM manually
                            if i==0:
                                get_weight_variable('W_zx',( fw_x_gru[0], fw_x_gru[1], 1, r_gru))
                                get_weight_variable('W_zh',( fw_h_gru[0], fw_h_gru[1], r_gru, r_gru))
                                get_weight_variable('W_rx',(fw_x_gru[0], fw_x_gru[1], 1, r_gru))
                                get_weight_variable('W_rh',(fw_h_gru[0], fw_h_gru[1], r_gru, r_gru))
                                get_weight_variable('W_hx',(fw_x_gru[0],fw_x_gru[1], 1, r_gru))
                                get_weight_variable('W_hh',(fw_h_gru[0], fw_h_gru[1], r_gru, r_gru)) 
        #                        get_weight_variable('TD_W', (DFm*r_lstm,dim_out))  
                            else:
                                get_weight_variable('W_zx',( fw_x_gru[0],fw_x_gru[1], r_gru, r_gru))
                                get_weight_variable('W_zh',( fw_h_gru[0], fw_h_gru[1], r_gru, r_gru))
                                get_weight_variable('W_rx',(fw_x_gru[0], fw_x_gru[1], r_gru, r_gru))
                                get_weight_variable('W_rh',(fw_h_gru[0], fw_h_gru[1],  r_gru, r_gru))
                                get_weight_variable('W_hx',(fw_x_gru[0], fw_x_gru[1],  r_gru, r_gru))
                                get_weight_variable('W_hh',(fw_h_gru[0], fw_h_gru[1],  r_gru, r_gru))    
                            get_weight_variable('alpha',(r_gru))                           
                
                # with tf.variable_scope('Merger'):
                #     for i, feature_dim in enumerate(self.num_gru_layers):   
                #         with tf.variable_scope('block{}'.format(i)):                    
                #             get_weight_variable('W', (self.batch_size, 2*r_gru, r_gru))
                #             get_bias_variable('b', (r_gru))  
                #             get_weight_variable('alpha',(r_gru))
                            
            with tf.variable_scope('TD_layer'):
                # final Fully connected layer
#                get_weight_variable('TD_W1', (DFm*r_lstm,fc_unit))
                # get_weight_variable('TD_W2', (DFm*2* r_gru,dim_out))
                if self.use_biases['TD_layer']:
                    get_bias_variable('TD', (dim_out))                           



    def parametric_relu(self,_x,r,ema=None):
#      r = self.n_channels  
      alphas = get_bias_variable('alpha', shape=( r), ema=ema)
      pos = tf.nn.relu(_x)
      neg = alphas * (_x - abs(_x)) * 0.5
    
      return pos + neg
      
    def TimeDistributed(self, X, ema=None):
        # pdb.set_trace()    
        DFm=X.shape[1].value
        r_gru=self.num_channel_gruCNN
        dim_out = self.masker_length+1
        X_ticks=tf.unstack(X,axis=2)
        out=[]
        with tf.variable_scope('TD_layer', reuse= tf.AUTO_REUSE):
            for X_t in X_ticks:
                X_t=tf.reshape(X_t,(self.batch_size,-1))
                TD_W2=get_weight_variable('TD_W2', (DFm*r_gru,dim_out))
                out_t2=tf.matmul(X_t,TD_W2)
                out_t2=tf.nn.relu(out_t2)
                out.append(out_t2)
        out=tf.transpose(tf.stack(out),(1,2,0)) 
        return out
      
        
    def gruCNN_forward(self,X,h_t0,index,P):
        # pdb.set_trace()

        fw_x=self.filter_size_gruCNN["X_filter"]["Layer{}".format(index+1)]
        fw_h=self.filter_size_gruCNN["H_filter"]["Layer{}".format(index+1)]
        r_gru = self.num_channel_gruCNN
        strd=self.stride["X_filter"]['Layer{}'.format(index+1)]
        if index==0:
            r=1
        else:
            r= r_gru
        with tf.variable_scope('Forward'):
            
            with tf.variable_scope('block{}'.format(index)):
                # z_t implement
                # pdb.set_trace() 
                W_zx=get_weight_variable('W_zx',(fw_x[0],fw_x[1], r, r_gru))
                W_zh=get_weight_variable('W_zh',(fw_h[0],fw_h[1], r_gru, r_gru))
                z_t=conv2D3(X,W_zx, strides=strd)+conv2D1(h_t0,W_zh,P)
                z_t=tf.sigmoid(z_t)
               
                # r_t implement
                W_rx=get_weight_variable('W_rx',(fw_x[0], fw_x[1], r, r_gru))
                W_rh=get_weight_variable('W_rh',(fw_h[0], fw_h[1], r_gru, r_gru))
                r_t=conv2D3(X,W_rx,strides=strd)+conv2D1(h_t0,W_rh,P)
                r_t=tf.sigmoid(r_t)
                
                # h_t_hat implementation
                W_hx=get_weight_variable('W_hx',(fw_x[0], fw_x[1], r, r_gru))
                W_hh=get_weight_variable('W_hh',(fw_h[0], fw_h[1], r_gru, r_gru))
                r_t_h_t=tf.multiply(r_t,h_t0[:,:,P:P+1,:])
                h_t_hat=conv2D3(X,W_hx,strides=strd)+conv2D2(r_t_h_t,W_hh)
                h_t_hat=tf.tanh(h_t_hat)
                
                
                h_t=tf.multiply(z_t,h_t0[:,:,P:P+1,:])+tf.multiply((1-z_t),h_t_hat)
                o_t=self.parametric_relu(h_t,r_gru)        

    #        h_t=tf.layers.dropout(h_t,rate=0.2)
        return o_t,o_t
        
    def gruCNN_backward(self,X,h_t0,index,P):
        # pdb.set_trace()

        fw_x=self.filter_size_gruCNN["X_filter"]["Layer{}".format(index+1)]
        fw_h=self.filter_size_gruCNN["H_filter"]["Layer{}".format(index+1)]
        r_gru = self.num_channel_gruCNN
        strd=self.stride["X_filter"]['Layer{}'.format(index+1)]
        if index==0:
            r=1
        else:
            r= r_gru
        with tf.variable_scope('Forward'):
            
            with tf.variable_scope('block{}'.format(index)):
                # z_t implement
                # pdb.set_trace() 
                W_zx=get_weight_variable('W_zx',(fw_x[0],fw_x[1], r, r_gru))
                W_zh=get_weight_variable('W_zh',(fw_h[0],fw_h[1], r_gru, r_gru))
                z_t=conv2D3(X,W_zx, strides=strd)+conv2D1(h_t0,W_zh,P)
                z_t=tf.sigmoid(z_t)
               
                # r_t implement
                W_rx=get_weight_variable('W_rx',(fw_x[0], fw_x[1], r, r_gru))
                W_rh=get_weight_variable('W_rh',(fw_h[0], fw_h[1], r_gru, r_gru))
                r_t=conv2D3(X,W_rx,strides=strd)+conv2D1(h_t0,W_rh,P)
                r_t=tf.sigmoid(r_t)
                
                # h_t_hat implementation
                W_hx=get_weight_variable('W_hx',(fw_x[0], fw_x[1], r, r_gru))
                W_hh=get_weight_variable('W_hh',(fw_h[0], fw_h[1], r_gru, r_gru))
                r_t_h_t=tf.multiply(r_t,h_t0[:,:,P:P+1,:])
                h_t_hat=conv2D3(X,W_hx,strides=strd)+conv2D2(r_t_h_t,W_hh)
                h_t_hat=tf.tanh(h_t_hat)
                
                
                h_t=tf.multiply(z_t,h_t0[:,:,P:P+1,:])+tf.multiply((1-z_t),h_t_hat)
                o_t=self.parametric_relu(h_t,r_gru)        

    #        h_t=tf.layers.dropout(h_t,rate=0.2)
        return o_t,o_t  
        
    def Concat(self, rnn_forw, rnn_back, index):
        # r_gru = self.num_channel_gruCNN    
        
        Y=[]
        with tf.variable_scope('Merger'):
            with tf.variable_scope('block{}'.format(index)):
#                W= get_weight_variable('W', (self.batch_size, 2*r_gru, r_gru))
#                b=get_bias_variable('b', (r_gru)) 
                for i in range(len(rnn_back)):
                    y_t= tf.concat([ rnn_forw[i], rnn_back[-(i+1)] ], axis=-1)
                    Y.append(y_t)
        return Y 
        
    def Adder(self, rnn_forw, rnn_back, index):
        r_gru = self.num_channel_gruCNN    
        
        Y=[]
        with tf.variable_scope('Merger'):
            with tf.variable_scope('block{}'.format(index)):
#                W= get_weight_variable('W', (self.batch_size, 2*r_gru, r_gru))
#                b=get_bias_variable('b', (r_gru)) 
                for i in range(len(rnn_back)):
                    y_t= tf.add(rnn_forw[i], rnn_back[-(i+1)])
                    Y.append(y_t)
        return Y        
        
    def BigruCNN_layer(self, layer_input,index,is_training=True,ema=None):
        r_gru=self.num_channel_gruCNN
        fw=self.filter_size_gruCNN["X_filter"]["Layer{}".format(index+1)]
        # L=layer_input.shape[2].value
        P_f=int((fw[0]-1)/2) # padd with the number of zeros frequency axis
        P_t=int((fw[1]-1)/2) # padd with the number of zeros temporal axis
        
        # pdb.set_trace()
        padding=tf.constant([[0,0],[P_f,P_f],[P_t,P_t],[0,0]])
        X=tf.pad(layer_input,padding, mode="CONSTANT")
        H=X.shape[1].value
        strd=self.stride["X_filter"]['Layer{}'.format(index+1)]
        DFm=np.floor((H-fw[0])/strd[0])+1
        # gruCNN manual implementation
        fw_h=self.filter_size_gruCNN["H_filter"]["Layer{}".format(index+1)]
        h_t0_f=tf.zeros(shape=(self.batch_size,DFm,fw_h[1],r_gru))
        h_t0_b=tf.zeros(shape=(self.batch_size,DFm,fw_h[1],r_gru))
        P_t=int((fw_h[1]-1)/2) # padd with the number of zeros temporal axis
#        X_flipped= tf.transpose(tf.reverse(tf.transpose(X, perm=[0,2,1,3]), axis=[1]), perm=[0,2,1,3])
        rnn_forw=[]
        rnn_back=[]
        with tf.variable_scope('BigruCNN_layer',reuse=True):
            for i in range(layer_input.shape[2].value):
                input_f=X[:,:,i:i+fw[1],:] # input to the forward network

                input_b=X[:,:,::-1,:][:,:,i:i+fw[1],:] # input to the backward network
                
                # pdb.set_trace()
                o_t_f, h_t_f= self.gruCNN_forward(input_f,h_t0_f,index,P_t)
                o_t_b, h_t_b= self.gruCNN_backward(input_b,h_t0_b,index, P_t)
                
                # modifying ht0
                h_t0_f=tf.unstack( tf.roll(h_t0_f,shift=-1, axis=2), axis=2)
                h_t0_f[P_t]=h_t_f[:,:,0,:]
                h_t0_f=tf.transpose(tf.stack(h_t0_f),perm=[1,2,0,3])
                
                h_t0_b=tf.unstack( tf.roll(h_t0_b,shift=-1, axis=2), axis=2)
                h_t0_b[P_t]=h_t_b[:,:,0,:]
                h_t0_b=tf.transpose(tf.stack(h_t0_b),perm=[1,2,0,3])
                        
#                o_t= tf.concat([o_t_f, o_t_b],-1)[:,:,P,:]   
                # weighted merging of the outputs

#                o_t=self.Merger(o_t, index)                
                rnn_forw.append(o_t_f[:,:,0,:])
                rnn_back.append(o_t_b[:,:,0,:])
                
            # pdb.set_trace() 
        # weighted merging
            rnn_outputs=self.Adder(rnn_forw, rnn_back, index)                
        rnn_outputs=tf.transpose(tf.stack(rnn_outputs),(1,2,0,3))

#        rnn_outputs=tf.nn.relu(rnn_outputs) 
        return rnn_outputs                                                                        #onput=[batch,max_time,op_length]
                                                                       #onput=[batch,max_time,op_length]

        
        
    
    def get_out_1_loss(self, Y_true, Y_pred):

        weight = self.cfg['loss']['out_1']['weight']
        l1_weight = self.cfg['loss']['out_1']['l1']
        l2_weight = self.cfg['loss']['out_1']['l2']


        if weight == 0:
            return Y_true * 0

        return weight * l1_l2_loss(Y_true, Y_pred, l1_weight, l2_weight)
    
    
    
    def inference(self, X, is_training, ema): 
        # Input X is mixed signal (clean_speech + noise) 
        with tf.variable_scope('BigruCNN_SE', reuse=True):
            #X -> gruCNN_layer0-> ... gruCNN_layerN 

            for i, dilation in enumerate(self.num_gru_layers):         
                X= self.BigruCNN_layer(X, i, is_training, ema)
            # Time Distributed layer
            clean_mask_pred= self.TimeDistributed(X, ema)
#            pdb.set_trace()
            clean_mask_pred=tf.expand_dims(clean_mask_pred,axis=3)
        return clean_mask_pred        
        
        
        
    def define_train_computations(self, optimizer, train_audio_conditions_reader, valid_audio_conditions_reader, global_step):
        # Train operations 
        self.train_audio_conditions_reader = train_audio_conditions_reader

        mixed_audio_train, clean_audio_train= train_audio_conditions_reader.dequeue()
        self.input=mixed_audio_train
        self.target=clean_audio_train
        
#        clean_audio_train = clean_audio_train[:, :, self.half_receptive_field:-self.half_receptive_field, :]  # target1
#        pdb.set_trace()
        clean_audio_pred = self.inference(mixed_audio_train, is_training=True, ema=None)
        self.predi=clean_audio_pred
        # Loss of train data (Time domain)
        self.train_loss = self.get_out_1_loss(clean_audio_train, clean_audio_pred)
        trainable_variables = tf.trainable_variables()
        # Loss of train dat (STFT domain)
#        pdb.set_trace()
#        # Regularization loss 
#        if self.l2 is not None:
#            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if not('_b' in v.name)])
#            self.train_loss += self.l2*l2_loss

        trainable_variables = tf.trainable_variables()
       
        optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm=5.0) # clipping the gradient
        
#        pdb.set_trace()
        self.gradients_update_op = optimizer.minimize(self.train_loss, global_step=global_step, var_list=trainable_variables)
#        optimizer = tf.train.AdamOptimizer(1e-3)

        if self.use_ema:
            self.maintain_averages_op = tf.group(self.ema.apply(trainable_variables)) 
            self.update_op = tf.group(self.gradients_update_op, self.maintain_averages_op)
        else:
            self.update_op = self.gradients_update_op

         

        # Validation operations
        self.valid_audio_conditions_reader = valid_audio_conditions_reader

        mixed_audio_valid, clean_audio_valid = valid_audio_conditions_reader.dequeue()

#        clean_audio_valid = clean_audio_valid[:, :, self.half_receptive_field:-self.half_receptive_field, :]   # target 1

        clean_audio_pred_valid = self.inference(mixed_audio_valid, is_training=False, ema=self.ema)
     

        # Loss of validation data
        self.valid_loss = self.get_out_1_loss(clean_audio_valid, clean_audio_pred_valid)        
        
      
    def train_epoch(self, coord, sess, logger):
        self.train_audio_conditions_reader.reset()
        thread = self.train_audio_conditions_reader.start_enqueue_thread(sess) 

        total_train_loss = 0
        total_batches = 0 
        
        while (not coord.should_stop()) and self.train_audio_conditions_reader.check_for_elements_and_increment():
            batch_loss, _ = sess.run([self.train_loss, self.update_op]) 
#            input, target, predi= sess.run([self.input,self.target,self.predi])
            
            
            if math.isnan(batch_loss):
                logger.critical('train cost is NaN')
                coord.request_stop() 
                break 
            total_train_loss += batch_loss
            total_batches += 1  
##            pdb.set_trace()  
#            axis=plt.subplot2grid([1,3],loc=[0,0])
#            ax=sns.heatmap(input[0,:,:,0], cmap="Reds", ax=axis)
#            plt.title('input image')
#            ax.invert_yaxis()
#            axis=plt.subplot2grid([1,3],loc=[0,1])
#            ax=sns.heatmap(target[0,:,:,0], cmap="Reds",ax=axis)    
#            plt.title('target image')
#            ax.invert_yaxis()
#            axis=plt.subplot2grid([1,3],loc=[0,2])
#            ax=sns.heatmap(predi[0,:,:,0], cmap="Reds",ax=axis)    
#            plt.title('target image')
#            ax.invert_yaxis()
#            plt.show()
            print( 'Batch loss is %s' %(batch_loss))
        coord.join([thread])
        
        if total_batches > 0:  
            average_train_loss = total_train_loss/total_batches 
        
        # Plots of features
        print( 'AVARAGE TRAIN LOSS IS %s' %(average_train_loss))

        return average_train_loss         
        
    
    def valid_epoch(self, coord, sess, logger):
        self.valid_audio_conditions_reader.reset()
        thread = self.valid_audio_conditions_reader.start_enqueue_thread(sess) 

        total_valid_loss = 0
        total_batches = 0 

        while (not coord.should_stop()) and self.valid_audio_conditions_reader.check_for_elements_and_increment():
            batch_loss = sess.run(self.valid_loss)
            if math.isnan(batch_loss):
                logger.critical('valid cost is NaN')
                coord.request_stop()
                break  
            total_valid_loss += batch_loss
            total_batches += 1  

        coord.join([thread])  

        if total_batches > 0:  
            average_valid_loss = total_valid_loss/total_batches  

        return average_valid_loss        
        
        
    def train(self, cfg, coord, sess):
        logger = logging.getLogger("msg_logger") 

        started_datestring = "{0:%Y-%m-%d, %H-%M-%S}".format(datetime.now())
        logger.info('Training of FFTNet started at: ' + started_datestring + ' using Tensorflow.\n')
        logger.info(get_info(cfg))

#        if self.use_batch_normalization and self.use_biases['filter_gate']:
#            print('Warning: Batch normalization should not be used in combination with filter and gate biases.')
#            logger.warning('Warning: Batch normalization should not be used in combination with filter and gate biases. Change the configuration file.')

        start_time = time.time()

        n_early_stop_epochs = cfg['n_early_stop_epochs']
        n_epochs = cfg['n_epochs']
       
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=4)

        early_stop_counter = 0

        min_valid_loss = sys.float_info.max
        epoch = 0
        while (not coord.should_stop()) and (epoch < n_epochs):
            epoch += 1
            epoch_start_time = time.time() 
            train_loss = self.train_epoch(coord, sess, logger) 
            valid_loss = self.valid_epoch(coord, sess, logger) 

            epoch_end_time = time.time()
                         
            info_str = 'Epoch=' + str(epoch) + ', Train: ' + str(train_loss) + ', Valid: '
            info_str += str(valid_loss) + ', Time=' + str(epoch_end_time - epoch_start_time)  
            logger.info(info_str)

            if valid_loss < min_valid_loss: 
                logger.info('Best epoch=' + str(epoch)) 
                save_variables(sess, saver, epoch, cfg, self.model_id) 
                min_valid_loss = valid_loss 
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter > n_early_stop_epochs:
                # too many consecutive epochs without surpassing the best model
                logger.debug('stopping early')
                break

        end_time = time.time()
        logger.info('Total time = ' + str(end_time - start_time))

        if (not coord.should_stop()):
            coord.request_stop()

    def define_generation_computations(self, i_shape): 
        self.noisy_audio_test = tf.placeholder(shape=i_shape, dtype=_FLOATX)
   
#        if self.lc_enabled:
#            self.lc_test = tf.placeholder(shape=(1, 1, None, self.label_dim), dtype=_FLOATX)  
#        else:
#            self.lc_test = None
#    
#        if self.gc_enabled:
#            self.gc_test = tf.placeholder(shape=(1, None), dtype=tf.int32) 
#        else:
#            self.gc_test = None   
       
        if self.use_ema:
            self.inference(self.noisy_audio_test, is_training=False, ema=None)
            self.ema.apply(tf.trainable_variables()) 

        self.clean_audio_pred_computation_graph = self.inference(self.noisy_audio_test, is_training=False, ema=self.ema) 
        self.test_loss = self.get_out_1_loss(self.noisy_audio_test, self.clean_audio_pred_computation_graph )


    def generation(self, sess, noisy_audio, clean_audio, cfg):
#        regain=self.cfg['regain']
#        rms_sig=util.rms(noisy_audio)
#        noisy_audio=(regain/rms_sig)*noisy_audio
#        pdb.set_trace()
        fft_noisy= self.data_segmentations_test(noisy_audio)
        fft_noisy=fft_noisy
        # input to the model 
        t=fft_noisy.shape[2]
        f=fft_noisy.shape[1]
        i_shape=(1,f,t,1)
        self.define_generation_computations(i_shape)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        restore_variables(sess, cfg)
        # placeholder=np.zeros((1,161,900,1))
        
        feature_input=10*np.log10(10**2*np.abs(fft_noisy)**2+1)
        # plt.matshow(feature_input[0,::-1,:,0])
        # plt.show()
#        inp_len=feature_input.shape[0]
        # placeholder[:,:,0:t,:]=feature_input
#        pdb.set_trace()      
        feed_dict = {self.noisy_audio_test:feature_input}
        
        model_out = sess.run(self.clean_audio_pred_computation_graph, feed_dict=feed_dict)
      #  test_loss=sess.run(self.test_loss,feed_dict={self.noisy_audio_test:feature_input,self.clean_audio_pred_computation_graph:model_out})
    #    print('test loss', test_loss)
        model_out=model_out;
        
        #### R_2 score over the frames  
#        clean_fft= self.data_segmentations_test(clean_audio)   
#        feature_target=10*np.log10(10**2*np.abs(clean_fft)**2+1)  
#        # pdb.set_trace()     
#        mean=np.mean(feature_target[0,::-1,:,0], axis=0)
#        idx=mean>1
#        chunk_target=feature_target[0,::-1,idx,0]
#        chunk_out=model_out[0,::-1,idx,0]

        # t_from=0
        # for i in range(10, t, 200):
        #     t_to=i
#        R_2=r2_score(chunk_target, chunk_out )   
#        print(R_2)
#        file=open('../R_2.txt', 'a')
#        file.write('\n'+str(R_2))
#        file.close()
        
            # t_from=t_to
        #####
            
        fft_mag=np.sqrt((10**(model_out/10)-1)/10**2)
#        pdb.set_trace()
        fft_mag=np.squeeze(fft_mag,axis=3)
        fft_mag=np.transpose(fft_mag,(0,2,1))
        fft_mag=np.reshape(fft_mag,(-1,161))      
        fft_noisy=np.squeeze(fft_noisy,axis=3)
        fft_noisy=np.transpose(fft_noisy,(0,2,1))
        fft_noisy=np.reshape(fft_noisy,(-1,161))  
#        pdb.set_trace() 
#        mask_input_batch=np.squeeze(mask_input_batch,axis=2)
        # signal reconstruction with noisy phase
        fft_mag_noisy=np.maximum(np.abs(fft_noisy), 1e-6)
        fft_phase=fft_noisy/fft_mag_noisy 
        clean_audio_segments=np.fft.irfft(np.multiply(fft_mag,fft_phase)).astype(np.float64)
        clean_audio_pred= self.overlap_add(clean_audio_segments)          
#        clean_audio_pred = clean_audio_segments.reshape((-1, ))
        noisy_audio_segments=np.fft.irfft(np.multiply(fft_mag_noisy,fft_phase)).astype(np.float64)
        noisy_audio_pred=self.overlap_add(noisy_audio_segments)
#        noisy_audio_pred = noisy_audio_segments.reshape((-1, ))        
        return clean_audio_pred, noisy_audio_pred
    
    
    
    
    def data_segmentations_test(self,noisy_audio):
        
        n_samples = noisy_audio.shape[0] 
        num_itr=int((n_samples-self.frame_size)/self.frame_shift)  
#        pdb.set_trace()
        noisy_audio_segments = stride_tricks.as_strided(noisy_audio,
                                                        shape=(num_itr,self.frame_size),
                                                        strides=(                                                                
                                                                noisy_audio.strides[0]*self.frame_shift,
                                                                noisy_audio.strides[0]))
#        win=np.hamming(self.frame_size)
#        noisy_audio_segments *=win
        
#        n=int(num_itr/128+1)
#        arr=np.zeros((n*128,self.frame_size))
#        arr[:noisy_audio_segments.shape[0],:noisy_audio_segments.shape[1]]=noisy_audio_segments
#        noisy_audio_segments=arr
#        noisy_audio_segments=np.reshape(noisy_audio_segments,(n,self.num_input_frames,self.frame_size))
        noisy_audio_segments=np.fft.rfft(noisy_audio_segments)
#        noisy_audio_segments=np.transpose(noisy_audio_segments,[0,2,1]) 
        noisy_audio_segments=np.transpose(noisy_audio_segments,[1,0]) 
        noisy_audio_segments=np.expand_dims(noisy_audio_segments,axis=0)
        noisy_audio_segments=np.expand_dims(noisy_audio_segments,axis=3)         
        return noisy_audio_segments
    
    def overlap_add(self,audio_segments):
#        pdb.set_trace()
        L_shift=(self.frame_shift)
        L_frame=self.frame_size
        L_sig=L_shift*audio_segments.shape[0]
        offsets = range(0, L_sig, L_shift)
        res = np.zeros(L_sig+L_shift, dtype=np.float64)
#        win=np.hamming(self.frame_size)
#        audio_segments *=win
        for i, n in enumerate(offsets):
            res[n:n+L_frame] += audio_segments[i,:]
            
        return res
