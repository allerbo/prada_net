import pandas as pd
import numpy as np
import sys

df = pd.read_csv('logs/fcts_08_55_06.txt',sep=';',names=['noise_seed','opt_seed', 'mse','fcts', 'n_nodes', 'n_links', 'lbda','gamma'])
#


#find lowest mse per noise_seed
df_opt = df.sort_values('mse').groupby(['noise_seed','lbda','gamma'],as_index=False).first()
df_opt = df_opt.drop(['opt_seed', 'mse'],axis=1)

#extract prada, lasso same compl, lasso same # fcts
prada=df_opt[df_opt.gamma==2].fcts
lasso29=df_opt[(df_opt.gamma==0) & (df_opt.lbda==2.9)].fcts
lasso13=df_opt[(df_opt.gamma==0) & (df_opt.lbda==1.3)].fcts


for fcts in (prada, lasso29, lasso13):
  fct_count={}
  #for each function, use function name as key and save total number of nodes as value in dict
  for counter in fcts:
    fct_vec= str(counter).split(',')
    fct_vec[0]=fct_vec[0][9:]
    fct_vec[-1]=fct_vec[-1][:-2]
    fct_vec = map(lambda f: f.strip(" '"),fct_vec)
    for name_nodes in fct_vec:
      name=name_nodes.split(':')[0].strip("'")
      nodes=int(name_nodes.split(':')[1])
      if not name in fct_count:
        fct_count[name]=[0,0]
         
      fct_count[name][0]+=1
      fct_count[name][1]+=nodes
     
  #for each fct in dict, write fct name and average # nodes.
  for fct in sorted(fct_count):
    covars = fct.split('-')
    name = '$f('
    for covar in covars:
      name+='x_'+str(int(covar)+1)+','
    name=name[:-1]+')$'
    print name + ' & - & - & ' + str(round(1.*fct_count[fct][0]/fcts.shape[0],2))+' & ' + str(round(1.*fct_count[fct][1]/fct_count[fct][0],2))+'\\\\'
  
  print ''

