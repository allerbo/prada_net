import numpy as np
import os, sys

sys.path.insert(1,'..')
from prada_net import prada


#Simulate data
np.random.seed(0)
x_mat = np.random.uniform(-1,1,(1000,5))
y_mat = x_mat[:,0] \
      + 0.5*(3*np.square(x_mat[:,1])-1) \
      + 0.5*(5*np.power(x_mat[:,2],3)-3*x_mat[:,2]) \
      + 0.125*(35*np.power(x_mat[:,3],4)-30*np.square(x_mat[:,3])+3)
y_mat = y_mat.reshape((len(y_mat),1))

#use optimal seeds for plot and graph
np.random.seed(24)
y_mat += np.random.normal(0,.1,y_mat.shape) #add noise

prada(x_mat=x_mat, y_mat=y_mat, TF_SEED=4, NP_SEED=4, BATCH_DIV=5, LOAD_APDX=None, DIM_H1=50, EPOCHS=[5000,5000,1000,0], LBDAS=[1.1, 1e-5],GAMMAS=[2.,2.], STEP_SIZE=1e-5, SIGMA_MAX=1e-2,PRINT=True)
os.system('bash ../copy_params.sh l1')
os.system('Rscript ../make_graph.R l1')
