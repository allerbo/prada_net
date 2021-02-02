import numpy as np
import os,sys
sys.path.insert(1,'..')
from prada_net import prada

LOG_NAME = sys.argv[1]
BOOT_SEED = int(sys.argv[2])
OPT_SEED = int(sys.argv[3])
if len(sys.argv)>4:
  LBDA = float(sys.argv[4])
  PHASE=2
else:
  PHASE=1

if 'SLURM_SUBMIT_DIR' in os.environ:
  slurm_root=os.environ['SLURM_SUBMIT_DIR']+'/'
else:
  slurm_root=''

#Simulate data
NOISE_LEVEL=0.1
np.random.seed(0)
x_mat = np.random.uniform(-1,1,(1000,5))
y_mat = x_mat[:,0] \
      + 0.5*(3*np.square(x_mat[:,1])-1) \
      + 0.5*(5*np.power(x_mat[:,2],3)-3*x_mat[:,2]) \
      + 0.125*(35*np.power(x_mat[:,3],4)-30*np.square(x_mat[:,3])+3)
y_mat = y_mat.reshape((len(y_mat),1))

y_mat_noise = y_mat + np.random.normal(0,NOISE_LEVEL,y_mat.shape)
log_seed = LOG_NAME+'_'+str(BOOT_SEED)

if PHASE==1:
  #first phase 
  prada(x_mat=x_mat, y_mat=y_mat_noise, NP_SEED=BOOT_SEED, TF_SEED=OPT_SEED, BATCH_DIV=5, LOAD_APDX=None, DIM_H1=50, EPOCHS=[5000,0,0,0], LBDAS=[0, 1e-5],GAMMAS=[2,2], STEP_SIZE=1e-5, PRINT=False, RETURN=False, LOG_NAME=log_seed)
  os.system('bash ../copy_params.sh phase1_'+log_seed+' '+log_seed)
else:
  #rest
  mse,_,_,_,_=prada(x_mat=x_mat, y_mat=y_mat_noise, NP_SEED=BOOT_SEED, TF_SEED=OPT_SEED, BATCH_DIV=5, LOAD_APDX='phase1_'+log_seed, DIM_H1=50, EPOCHS=[00,2000,300,0], LBDAS=[LBDA, 1e-5],GAMMAS=[2,2], STEP_SIZE=1e-5, PRINT=False, RETURN=True, LOG_NAME=log_seed)
  with open(slurm_root+'logs/'+LOG_NAME+'.txt', 'a+') as f:
    f.write(';'.join(map(str,(BOOT_SEED, OPT_SEED, LBDA, mse))) + '\n')
