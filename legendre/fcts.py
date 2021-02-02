import numpy as np
import pandas as pd
import os,sys
sys.path.insert(1,'..')
from prada_net import prada

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
epochs2=[0, 5000, 1000, 0]
dim_h1=50
lbda1 = 1.1 #lbda for prada
lbda2 = 1.3 #lbda for lasso with same complexity
lbda3 = 2.9 #lbda for lasso with same #fcts

x_mat = np.random.uniform(-1,1,(1000,5))
y_mat = x_mat[:,0] \
    + 0.5*(3*np.square(x_mat[:,1])-1) \
    + 0.5*(5*np.power(x_mat[:,2],3)-3*x_mat[:,2]) \
    + 0.125*(35*np.power(x_mat[:,3],4)-30*np.square(x_mat[:,3])+3)
y_mat = y_mat.reshape((len(y_mat),1))

np.random.seed(NOISE_SEED)
y_mat_noise = y_mat + np.random.normal(0,NOISE_LEVEL,y_mat.shape)
log_seed = LOG_NAME+'_'+str(NOISE_SEED)

#Run phase one of prada
prada(x_mat=x_mat, y_mat=y_mat_noise, NP_SEED=OPT_SEED, TF_SEED=OPT_SEED, BATCH_DIV=5, LOAD_APDX=None, DIM_H1=dim_h1, EPOCHS=epochs1, LBDAS=[0, 0],GAMMAS=[0,0], STEP_SIZE=0,PRINT=False,LOG_NAME=log_seed,SAVE_XY=True)
os.system('bash ../copy_params.sh phase1_'+log_seed+' '+log_seed)

#Run step 2 and 3 for prada, and for the two lasso versions. (GAMMA=0 means standard lasso.)
GAMMA=2; LBDA=lbda1
mse,_,fcts,n_nodes,n_links=prada(x_mat=x_mat, y_mat=y_mat_noise, NP_SEED=OPT_SEED, TF_SEED=OPT_SEED, BATCH_DIV=5, LOAD_APDX='phase1_'+log_seed, DIM_H1=dim_h1, EPOCHS=epochs2, LBDAS=[LBDA, 1e-5],GAMMAS=[GAMMA,2], STEP_SIZE=1e-5,PRINT=False,RETURN=True,LOG_NAME=log_seed, SIGMA_MAX=1e-2)
with open(slurm_root+'logs/'+LOG_NAME+'.txt', 'a+') as f:
  f.write(';'.join(map(str,(NOISE_SEED, OPT_SEED, mse, fcts, n_nodes, n_links, LBDA, GAMMA))) + '\n')

GAMMA=0; LBDA=lbda2
mse,_, fcts,n_nodes,n_links=prada(x_mat=x_mat, y_mat=y_mat_noise, NP_SEED=OPT_SEED, TF_SEED=OPT_SEED, BATCH_DIV=5, LOAD_APDX='phase1_'+log_seed, DIM_H1=dim_h1, EPOCHS=epochs2, LBDAS=[LBDA, 1e-5],GAMMAS=[GAMMA,2], STEP_SIZE=1e-5,PRINT=False,RETURN=True,LOG_NAME=log_seed)
with open(slurm_root+'logs/'+LOG_NAME+'.txt', 'a+') as f:
  f.write(';'.join(map(str,(NOISE_SEED, OPT_SEED, mse, fcts, n_nodes, n_links, LBDA, GAMMA))) + '\n')

GAMMA=0; LBDA=lbda3
mse,_, fcts,n_nodes,n_links=prada(x_mat=x_mat, y_mat=y_mat_noise, NP_SEED=OPT_SEED, TF_SEED=OPT_SEED, BATCH_DIV=5, LOAD_APDX='phase1_'+log_seed, DIM_H1=dim_h1, EPOCHS=epochs2, LBDAS=[LBDA, 1e-5],GAMMAS=[GAMMA,2], STEP_SIZE=1e-5,PRINT=False,RETURN=True,LOG_NAME=log_seed)
with open(slurm_root+'logs/'+LOG_NAME+'.txt', 'a+') as f:
  f.write(';'.join(map(str,(NOISE_SEED, OPT_SEED, mse, fcts, n_nodes, n_links, LBDA, GAMMA))) + '\n')

