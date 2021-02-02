import numpy as np
import tensorflow as tf
import os,sys

def dgr(LOAD_APDX, LBDA, PRINT=False,SEED=None):
  def print1(print_str):
    if 'SLURM_SUBMIT_DIR' in os.environ:
       with open(os.environ['SLURM_SUBMIT_DIR']+'/drp_out'+TIME_STR+'.txt', 'a+') as f:
         f.write(str(print_str) + '\n')
    else:
      print(print_str)
  
  def savetxt1(name, var):
    if 'SLURM_SUBMIT_DIR' in os.environ:
      np.savetxt('data/'+name+TIME_STR+'.txt', var)
    else:
      np.savetxt('data/'+name+'.txt', var)
  
  
  def set_shape(a, b=False):
    if b and len(a.shape) == 0:
      shp = (1,)
    elif not b and len(a.shape) == 1:
      shp = (a.shape[0],1)
    else:
      shp = a.shape
    return shp
  
  
  def limit(x,x_min,x_max):
    return tf.maximum(x_min,tf.minimum(x_max,x))
  
  np.set_printoptions(suppress=True)
  
  if not SEED is None:
    np.random.seed(SEED)
    tf.set_random_seed(SEED)
  else:
    np.random.seed()
  
  W1 = tf.constant(np.loadtxt('data/W1_'+LOAD_APDX+'.txt'),dtype=tf.float32)
  b1 = tf.constant(np.loadtxt('data/b1_'+LOAD_APDX+'.txt'),dtype=tf.float32)
  W2_np = np.loadtxt('data/W2_'+LOAD_APDX+'.txt')
  b2_np = np.loadtxt('data/b2_'+LOAD_APDX+'.txt')
  W2 = tf.constant(W2_np.reshape(set_shape(W2_np)),dtype=tf.float32)
  b2 = tf.constant(b2_np.reshape(set_shape(b2_np,True)),dtype=tf.float32)
  DIM_H1=int(W2.get_shape()[0])
  
  x_train = np.loadtxt('data/x_train_'+LOAD_APDX+'.txt')
  x_test = np.loadtxt('data/x_test_'+LOAD_APDX+'.txt')
  y_train = np.loadtxt('data/y_train_'+LOAD_APDX+'.txt')
  y_test = np.loadtxt('data/y_test_'+LOAD_APDX+'.txt')
  y_train = y_train.reshape((len(y_train),1))
  y_test = y_test.reshape((len(y_test),1))
  
  x_in = tf.placeholder(tf.float32, [None, x_train.shape[1]]) #Placeholder for x_train/x_test
  y_in = tf.placeholder(tf.float32, [None, y_train.shape[1]]) #Placeholder for y_train/y_test
  
  
  beta1=tf.Variable(np.ones(DIM_H1), dtype=tf.float32)
  W2_beta = tf.matmul(tf.diag(limit(beta1,-1.,1.)),W2)
  
  h1 = tf.nn.tanh(tf.matmul(x_in, W1)+ b1)
  y_out = tf.matmul(h1, W2)+ b2
  y_out_beta = tf.matmul(h1, W2_beta)+ b2
  
  
  error_rec_opt = 1./2.*tf.reduce_mean(tf.square(y_out-y_out_beta))
  error_rec_ref = 1./2.*tf.reduce_mean(tf.square(y_in-y_out_beta))
  error_l1 = tf.reduce_sum(tf.abs(beta1))
  
  #Set up Tensorflow
  sess = tf.InteractiveSession()
  train_step = tf.train.AdamOptimizer().minimize(error_rec_opt+LBDA*error_l1)
  tf.global_variables_initializer().run() #initalize variables
  
  for epoch in range(10000):
    sess.run(train_step,feed_dict={x_in:x_train,y_in:y_train})
    if epoch % 1000 == 0:
      if PRINT:
        print ('error_opt',error_rec_opt.eval({x_in: x_train, y_in: y_train}))
        print ('error_ref',error_rec_ref.eval({x_in: x_test, y_in: y_test}))
        print ('error l1',error_l1.eval())
        nodes = np.sum(1*(np.abs(beta1.eval()>1e-3)))
      
        print ('nodes',nodes)
        print ''
      #savetxt1('W2_beta', W2_beta.eval())
  mse=error_rec_ref.eval({x_in: x_test, y_in: y_test})
  nodes = [np.sum(1*(np.abs(beta1.eval()>1e-3)))]#threshold
  params = [np.sum(np.abs(W1.eval())>1e-3), np.sum(np.abs(W2_beta.eval())>1e-3)]#threshold
  
  abs_grads = np.mean(np.abs(sess.run(tf.gradients(y_out_beta,x_in),{x_in: x_test})),1).tolist()
  sess.close()
  return (mse, abs_grads)
