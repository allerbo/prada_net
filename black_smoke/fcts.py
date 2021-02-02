import numpy as np
import pandas as pd
import os,sys
sys.path.insert(1,'..')
from prada_net import prada

print sys.argv
LOG_NAME = sys.argv[1]
NOISE_SEED = int(sys.argv[2])
OPT_SEED = int(sys.argv[3])

if 'SLURM_SUBMIT_DIR' in os.environ:
  slurm_root=os.environ['SLURM_SUBMIT_DIR']+'/'
else:
  slurm_root=''


#Simulate data
NOISE_LEVEL=0.1
np.random.seed(0)
epochs1=[5000, 0, 0, 0]
epochs2=[0, 1000, 300, 0]
dim_h1=100

GAMMA=2; LBDA=0.017 #prada
GAMMA=0; LBDA=0.9 #lasso

N_X=10000
DIM_X=11
rho = np.loadtxt('data/sp_rho.txt')
x_mat = np.random.multivariate_normal(np.zeros(DIM_X),rho,N_X)

APDX = 'bs_07_49_16_21'

W1 = np.loadtxt('data/W1_'+APDX+'.txt')
W2 = np.loadtxt('data/W2_'+APDX+'.txt')
W2 = W2.reshape((W2.shape[0],1))
b1 = np.loadtxt('data/b1_'+APDX+'.txt')
b2 = np.loadtxt('data/b2_'+APDX+'.txt')
b2 = b2.reshape((1,))

h1 = np.tanh(np.matmul(x_mat, W1)+ b1)
y_mat = np.matmul(h1, W2)+ b2

np.random.seed(NOISE_SEED)
y_mat_noise = y_mat + np.random.normal(0,NOISE_LEVEL,y_mat.shape)
log_seed = LOG_NAME+'_'+str(NOISE_SEED)

prada(x_mat=x_mat, y_mat=y_mat_noise, NP_SEED=OPT_SEED, TF_SEED=OPT_SEED, BATCH_DIV=5, LOAD_APDX=None, DIM_H1=dim_h1, EPOCHS=epochs1, LBDAS=[0, 0],GAMMAS=[0,0], STEP_SIZE=0,PRINT=False,LOG_NAME=log_seed,SAVE_XY=False)
os.system('bash ../copy_params.sh phase1_'+log_seed+' '+log_seed)

mse,_,fcts,n_nodes,n_links=prada(x_mat=x_mat, y_mat=y_mat_noise, NP_SEED=OPT_SEED, TF_SEED=OPT_SEED, BATCH_DIV=5, LOAD_APDX='phase1_'+log_seed, DIM_H1=dim_h1, EPOCHS=epochs2, LBDAS=[LBDA, 1e-5],GAMMAS=[GAMMA,2], STEP_SIZE=1e-5,PRINT=False,RETURN=True,LOG_NAME=log_seed, SIGMA_MAX=1e-2)
with open(slurm_root+'logs/'+LOG_NAME+'.txt', 'a+') as f:
  f.write(';'.join(map(str,(NOISE_SEED, OPT_SEED, mse, fcts, n_nodes, n_links, LBDA, GAMMA))) + '\n')

#Line below used to save parameters of optimal initialization for each noise seed
os.system('bash ../copy_params.sh phase2_'+log_seed+' '+log_seed)
