import pandas as pd
import numpy as np
import sys

#df = pd.read_csv('logs/fcts_16_17_12.txt',sep=';',names=['noise_seed','opt_seed', 'mse','fcts', 'n_nodes', 'n_links    ', 'lbda','gamma']) #lasso
df = pd.read_csv('logs/fcts_18_00_35.txt',sep=';',names=['noise_seed','opt_seed', 'mse','fcts', 'n_nodes', 'n_links    ', 'lbda','gamma']) #prada

df_opt = df.sort_values('mse').groupby(['noise_seed','lbda','gamma',],as_index=False).first()
df_opt = df_opt.drop(['opt_seed', 'mse'],axis=1)
fcts=df_opt.fcts

df_opt1 = df.sort_values('mse').groupby(['noise_seed','lbda','gamma',],as_index=False).first()

fct_freq={}
n_fcts=[]
for counter in fcts:
  fct_vec= str(counter).split(',')
  fct_vec[0]=fct_vec[0][9:]
  fct_vec[-1]=fct_vec[-1][:-2]
  n_fcts.append(len(fct_vec))
  for name_nodes in fct_vec:
    name=name_nodes.split(':')[0].strip(" '")
    nodes=int(name_nodes.split(':')[1])
    if not name in fct_freq:
      fct_freq[name]=[0,0]
       
    fct_freq[name][0]+=1
    fct_freq[name][1]+=nodes


#print('mean n_fcts: ',np.mean(n_fcts))
x_types=['\\texttt{y}','\\texttt{doy}','\\texttt{dow}','\\texttt{T}^\\texttt{0}','\\texttt{T}^\\texttt{1}','\\bar{\\texttt{T1}}', '\\bar{\\texttt{T2}}','\\texttt{r}','\\texttt{e}','\\texttt{n}','\\texttt{h}','\\alpha']

for fct in sorted(sorted(fct_freq),key=len):
  covars = fct.replace("'","").split('-')
  name = '$f('
  for covar in covars:
    name += x_types[int(covar)]+','
  name=name[:-1]+')$'
  if 1.*fct_freq[fct][0]/fcts.shape[0]>0.2:
    print name + ' & ' + str(round(1.*fct_freq[fct][0]/fcts.shape[0],2))+' & ' + str(round(1.*fct_freq[fct][1]/fct_freq[fct][0],2))+'\\\\'

print ''

