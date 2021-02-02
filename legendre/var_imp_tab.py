import pandas as pd
import numpy as np
import sys,os

os.system("sed -i 's/;/,/g' logs/var_imp_08_54_54.txt")
df = pd.read_csv('logs/var_imp_08_54_54.txt',sep=',',names=['alg','noise_seed','opt_seed', 'mse','x1', 'x2', 'x3', 'x4', 'x5', 'lbda','gamma'])
df.mse=pd.to_numeric(df.mse)
df.x1=pd.to_numeric(df.x1.str[3:])
df.x5=pd.to_numeric(df.x5.str[:-3])

#select lowest mse over optimization seeds
df_opt = df.groupby(['alg','noise_seed','x1','x2','x3','x4','x5'],as_index=False)['mse'].min()

means=df_opt.groupby('alg').mean().transpose()
stds=df_opt.groupby('alg').std().transpose()
means= means.round(2).applymap(str)
stds= stds.round(2).applymap(str)

dgr_text = means.dgr.str.cat(stds.dgr,sep=' $\\pm$ ')
lasso_text = means.lasso.str.cat(stds.lasso,sep=' $\\pm$ ')
prada_text = means.prada.str.cat(stds.prada,sep=' $\\pm$ ')

df_out = pd.concat([prada_text, lasso_text, dgr_text],axis=1,join='inner')
df_out=df_out.drop(['noise_seed'])
mse_row=df_out.loc['mse']
print 'Test error & - & '+mse_row[0]+' & '+mse_row[1]+' & '+mse_row[2]+'\\\\'

i=0
true_vals=[1, 1.5, 1.89, 2.23,0]
for index,row in df_out.drop(['mse']).iterrows():
  print '$x_'+index[-1]+'$ & '+str(true_vals[i])+' & '+row[0]+' & '+row[1]+' & '+row[2]+'\\\\'
  i+=1
