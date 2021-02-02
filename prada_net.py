def prada(x_mat, y_mat, TF_SEED, NP_SEED, BATCH_DIV, LOAD_APDX, DIM_H1, EPOCHS, LBDAS, GAMMAS, STEP_SIZE, LOG_NAME='', PRINT=True, RETURN=False, SAVE_XY=False, SIGMA_MAX=0):
  import numpy as np
  import tensorflow as tf
  from collections import Counter
  import os
  import scipy.sparse as sp


  #Print to logfile if on cluster, else to just print
  def print1(print_str):
    if 'SLURM_SUBMIT_DIR' in os.environ:
       with open(os.environ['SLURM_SUBMIT_DIR']+'/logs/'+LOG_NAME+'.txt', 'a+') as f:
         f.write(str(print_str) + '\n')
    else:
      print(print_str)
  
  #Save text with appendix
  def savetxt1(name, var):
    np.savetxt('data/'+name+LOG_NAME+'.txt', var)

  #Set correct shape on 1d numpy arrays
  def set_shape(a, b=False):
    if b and len(a.shape) == 0:
      shp = (1,)
    elif not b and len(a.shape) == 1:
      shp = (a.shape[0],1)
    else:
      shp = a.shape
    return shp

  def count_fcts(W, threshold):
    l1 = []
    for c in range(W.shape[1]):
      l1.append([x for x in np.where(np.abs(W[:,c])>threshold)[0]])
    l2 = map(lambda a: [11 if x>=11 else x for x in a],l1) #replace everything above 11 with 11
    l3 = map(sorted,map(list,map(set,l2))) #remove duplicate 11's
    l4 = map(lambda a: '-'.join(str(x) for x in a),l3) # list to string
    return len(set(l4).difference({''})) #count unique strings (excluding empty string)

  def get_fcts(W):
    l1 = []
    for c in range(W.shape[1]):
      l1.append([x for x in np.where(np.abs(W[:,c])>0)[0]])
    l2 = map(lambda a: '-'.join(str(x) for x in a),l1) # list to string
    l3 = filter(lambda a: a != '', l2) # remove empty nodes
    return Counter(l3)

  if not NP_SEED is None:
    np.random.seed(NP_SEED)
  else:
    np.random.seed()
  
  if not TF_SEED is None:
    tf.set_random_seed(TF_SEED)

  np.set_printoptions(suppress=True)
  DIM_X = x_mat.shape[1] #Number of x-dimensions
  DIM_Y = y_mat.shape[1] #Number of y-dimensions
  
  N_SAMPLES = y_mat.shape[0]
  assert N_SAMPLES == x_mat.shape[0]
  
  #Split in test and training data (95%  train and 5% test)
  n_train = int(round(0.95 * N_SAMPLES))
  p = np.random.permutation(N_SAMPLES)
  x_train, x_test = x_mat[p][:n_train,:], x_mat[p][n_train:,:] 
  y_train, y_test = y_mat[p][:n_train,:], y_mat[p][n_train:,:] 
  BATCH_SIZE = int(round(1.*x_train.shape[0]/BATCH_DIV)) #Number of samples used in each gradient descent update
  
  #USED in DGR
  if SAVE_XY:
    savetxt1('x_train_', x_train)
    savetxt1('x_test_', x_test)
    savetxt1('y_train_', y_train)
    savetxt1('y_test_', y_test)
  
  
  #Set up Tensorflow
  SESS = tf.InteractiveSession()
  x_in = tf.placeholder(tf.float32, [None, DIM_X]) #Placeholder for x_train/x_test
  y_in = tf.placeholder(tf.float32, [None, DIM_Y]) #Placeholder for y_train/y_test
  epoch_pl = tf.placeholder(tf.float32)
  
  #Four phases of the algorithm:
  #1. Train without regularization
  #2. Train with l1-regularization and standard optimizer
  #3. Train with l1-regularization and proximal gradient descent
  #4. Fix zero parameters and update non-zero parameters without regularization.
  #Parameters can either be loaded or initialized from scratch.
  for phase in range(4):
    if phase == 0:
      if not LOAD_APDX is None:
        W1 = tf.Variable(np.loadtxt('data/W1_'+LOAD_APDX+'.txt'),dtype=tf.float32)
        b1 = tf.Variable(np.loadtxt('data/b1_'+LOAD_APDX+'.txt'),dtype=tf.float32)
        W2_np = np.loadtxt('data/W2_'+LOAD_APDX+'.txt')
        W2 = tf.Variable(W2_np.reshape(set_shape(W2_np)),dtype=tf.float32)
        b2_np = np.loadtxt('data/b2_'+LOAD_APDX+'.txt')
        b2 = tf.Variable(b2_np.reshape(set_shape(b2_np,True)), dtype=tf.float32)

      else:
        W1 = tf.Variable(tf.random_uniform(shape=[DIM_X, DIM_H1],minval=-1.*np.sqrt(6./(DIM_X+DIM_H1)), maxval=1.*np.sqrt(6./(DIM_X+DIM_H1)))) 
        b1 = tf.Variable(np.zeros((DIM_H1)), dtype=tf.float32)
        W2 = tf.Variable(np.zeros((DIM_H1, DIM_Y)), dtype=tf.float32)
        b2 = tf.Variable(np.zeros((DIM_Y)), dtype=tf.float32)
      
      #Used in last phase for keeping some parameters zero, in other steps just a matrix of ones.
      W1_01 = np.ones(list(map(lambda x: x.value, W1.shape.dims)))
      W2_01 = np.ones(list(map(lambda x: x.value, W2.shape.dims)))

    else: #phase is 1 or 2 or 3
      if phase == 1:
        #save current weights and use in adaptive lasso
        init_weights_W1 = np.abs(W1.eval())
        init_weights_W2 = np.abs(W2.eval())
        lbda_l1_mat_W1_a = LBDAS[0]/np.maximum(1e-10,np.power(init_weights_W1,GAMMAS[0]))
        lbda_l1_mat_W2_a = LBDAS[0]/np.maximum(1e-10,np.power(init_weights_W2,GAMMAS[0]))
      if phase == 2: 
        #save current weights and use in adaptive lasso with proximal gradient descent
        init_weights_W1 = np.abs(W1.eval())
        init_weights_W2 = np.abs(W2.eval())
        lbda_l1_mat_W1_p = LBDAS[1]/np.power(np.maximum(1e-10,init_weights_W1),GAMMAS[1])
        lbda_l1_mat_W2_p = LBDAS[1]/np.power(np.maximum(1e-10,init_weights_W2),GAMMAS[1])
      if phase == 3:
        #remember zeros and keep them 0
        W1_01 = 1.*(np.abs(W1.eval())>0.)
        W2_01 = 1.*(np.abs(W2.eval())>0.)

      #Technical reason
      W1 = tf.Variable(W1.eval())
      b1 = tf.Variable(b1.eval())
      W2 = tf.Variable(W2.eval())
      b2 = tf.Variable(b2.eval())
     # end

      if phase == 1:
        error_l1 = tf.reduce_mean(tf.multiply(lbda_l1_mat_W1_a,tf.abs(W1)))
        error_l1 += tf.reduce_mean(tf.multiply(lbda_l1_mat_W2_a,tf.abs(W2)))

    #end if phase

    #Layer h1 as function of x_train/x_test, W1 and b1
    x_beta=0 #this is "y_hat before adding random effect"
    W1_t_01 = tf.multiply(W1, W1_01)
    W2_t_01 = tf.multiply(W2, W2_01)
    h1 = tf.nn.tanh(tf.matmul(x_in, W1_t_01)+ b1)
  
    x_beta += tf.matmul(h1, W2_t_01)+ b2
    y_hat = x_beta

    error_rec = tf.reduce_mean(tf.square(y_in-y_hat))# penalize difference between estimated y and real y
    #R squared
    y_mean = tf.reduce_mean(y_in,0)
    ss_tot = tf.reduce_mean(tf.square(y_in-tf.reduce_mean(y_in)))
    r_squared = 1-error_rec/ss_tot
    
    if phase == 0:
      train_step = tf.train.AdamOptimizer().minimize(error_rec)
    elif phase == 1:
      train_step = tf.train.AdamOptimizer().minimize(error_rec + error_l1)
    elif phase == 2:
      train_step = tf.train.GradientDescentOptimizer(STEP_SIZE).minimize(error_rec) #proximal step added later
    elif phase == 3:
      train_step = tf.train.AdamOptimizer().minimize(error_rec)
  

    tf.global_variables_initializer().run() #initalize variables
  
  
    for epoch in range(EPOCHS[phase]):
      #run for mini-batches
      for batch in range(-(-x_train.shape[0] // BATCH_SIZE)): #two minus signs to get ceiling division (instead of floor)
        start = batch * BATCH_SIZE
        stop = (batch + 1) * BATCH_SIZE
        x_batch = x_train[start:stop]
        y_batch = y_train[start:stop]
        SESS.run(train_step, feed_dict={x_in: x_batch, y_in: y_batch, epoch_pl: epoch}) #Do one optimization update with the batch data
        if phase == 2:
          #Proximal Gradient Descent step
          W1_np = W1.eval()
          W1_np = np.multiply(np.sign(W1_np),np.maximum(np.abs(W1_np)-STEP_SIZE*lbda_l1_mat_W1_p,0.))
          W1.load(W1_np)
          W2_np = W2.eval()
          W2_np = np.multiply(np.sign(W2_np),np.maximum(np.abs(W2_np)-STEP_SIZE*lbda_l1_mat_W2_p,0.))
          W2.load(W2_np)
  
      if epoch % 100 == 0:
        if PRINT:
          print1(('Phase: ',phase))
          print1(('Epoch: ',epoch))
          print1(('Error batch: ', error_rec.eval({x_in: x_batch, y_in: y_batch})))
          print1(('Error test: ', error_rec.eval({x_in: x_test, y_in: y_test})))
          print1(('R^2 test: ', r_squared.eval({x_in: x_test, y_in: y_test})))
          print1((np.sum(np.abs(W1.eval())>0.001), np.sum(np.abs(W1.eval())>0)))
          print1((np.sum(np.abs(W2.eval())>0.001), np.sum(np.abs(W2.eval())>0)))
          fcts_1=count_fcts(W1.eval(),0.001)
          fcts_2=count_fcts(W1.eval(),0.)
          print1((fcts_1, fcts_2))

        savetxt1('W1_', W1.eval())
        savetxt1('W2_', W2.eval())
        savetxt1('b1_', b1.eval())
        savetxt1('b2_', b2.eval())

    if PRINT:
      fcts_1=count_fcts(W1.eval(),0.001)
      fcts_2=count_fcts(W1.eval(),0.)
      print1((fcts_1, fcts_2))
      print1(get_fcts(W1.eval()))

    #Identify linear functions
    if phase==2 and SIGMA_MAX>0:
      W1_np= W1.eval()
      W2_np= W2.eval()
      b1_np= b1.eval()
      W1_out = np.copy(W1_np)
      b1_out = np.copy(b1_np)
      W2_out = np.copy(W2_np)
    
      fct_nodes={}
      nodes = np.where(np.sum(np.abs(W1_np),0)>0)[0] #hidden nodes in use
      for node in nodes: #for each used hidden node, identify its connected input nodes
        fct = np.where(np.abs(W1_np[:,node])>0)[0].tolist()
        fct_str='-'.join(map(str,fct))
        if not fct_str in fct_nodes.keys():
          fct_nodes[fct_str] = [node]
        else:
          fct_nodes[fct_str].append(node)
          
      #fct_nodes is a dict with fct, defined by its intputs, as keys, and a list of used hidden nodes an values.
      #Calculate partial derivatives for each function
      for fct in fct_nodes.keys(): 
        nodes = fct_nodes[fct]
        #W1_sub is a fct specific copy of W1, where all links not used to realize fct are set to 0.
        W1_sub = np.copy(W1_np)
        delete_nodes = range(W1_sub.shape[1])
        for node in nodes:
          delete_nodes.remove(node)
         
        W1_sub[:,delete_nodes]=0
        W1_sub_tf = tf.Variable(W1_sub,dtype=tf.float32)
        tf.global_variables_initializer().run() #initalize variables
  
        sum_x=0 #first moments, to be used when calculating variance of derivative
        sum_x2=0 #second moments, to be used when calculating variance of derivative
        for batch in range(-(-x_train.shape[0] // BATCH_SIZE)): #two minus signs to get ceiling division (instead of floor)
          start = batch * BATCH_SIZE
          stop = (batch + 1) * BATCH_SIZE
          x_batch = x_train[start:stop]
          y_out = tf.matmul(tf.nn.tanh(tf.matmul(x_in,W1_sub_tf)+b1),W2)+b2
          
          derivs = SESS.run(tf.gradients(y_out,x_in),{x_in: x_batch})
          sum_x+= np.sum(derivs,1)
          sum_x2+= np.sum(np.square(derivs),1)
          
  
        n=1.*x_train.shape[0]
        der_var=(sum_x2-np.square(sum_x)/n)/(n-1) #empirical variance
  
        #for each identified linear covariate, move its connections from existing node(s) to a new one.
        for linear in list(set(map(int,fct.split('-'))).intersection(set(np.where(der_var<SIGMA_MAX)[0].tolist()))):
          for node in nodes:
            W1_out[linear,node]=0#remove connection for linear fct
            new_node_W1 = np.zeros((W1_np.shape[0],1))
            new_node_W1[linear,0]=W1_np[linear,node]#copy W1 connection to new linear node
            W1_out=np.hstack((W1_out,new_node_W1))
            b1_out=np.hstack((b1_out,[0]))
            new_node_W2 = np.zeros((W1_np.shape[0],1))
            new_node_W2=W2_np[node,0]#copy W2 connection to new linear node
            W2_out = np.vstack((W2_out,new_node_W2))
   
          
      if PRINT:
        print1((np.sum(np.abs(W1_out)>0.001), np.sum(np.abs(W1_out)>0)))
        print1((np.sum(np.abs(W2_out)>0.001), np.sum(np.abs(W2_out)>0)))
        fcts_1=count_fcts(W1_out,0.001)
        fcts_2=count_fcts(W1_out,0.)
        print1((fcts_1, fcts_2))
        print1(get_fcts(W1_out))
  
  
      savetxt1('W1_', W1_out)
      savetxt1('W2_', W2_out)
      savetxt1('b1_', b1_out)
      savetxt1('b2_', b2.eval()) #unaffected
      W1 = tf.Variable(W1_out,dtype=tf.float32)
      W2 = tf.Variable(W2_out,dtype=tf.float32)
      b1 = tf.Variable(b1_out,dtype=tf.float32)
      tf.global_variables_initializer().run() #initalize variables
    

  if RETURN:
    mse = error_rec.eval({x_in: x_test, y_in: y_test})
    abs_grads = np.mean(np.abs(SESS.run(tf.gradients(y_hat,x_in),{x_in: x_test})),1).tolist()
    fcts = get_fcts(W1.eval())
    n_nodes = (np.sum(np.sum(np.abs(W1.eval())>0,0)>0), np.sum(np.sum(np.abs(W2.eval())>0,1)>0))
    n_links = np.sum(np.abs(W1.eval())>0)+ np.sum(np.abs(W2.eval())>0)
    return mse, abs_grads, fcts, n_nodes, n_links
