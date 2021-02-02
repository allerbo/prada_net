import pandas as pd
import numpy as np
#from matplotlib import pyplot as plt
import sys
FS=12
df = pd.read_csv('logs/lbda_sweep_10_16_59.txt',sep=';',names=['boot_seed','opt_seed', 'lbda','mse'])

#Find lowest mse per bootstrap and lambda
df_opt = df.groupby(['boot_seed', 'lbda'],as_index=False)['mse'].min()

#Calculate mean +- 2 standard deviations for each lambda value
means=df_opt.groupby('lbda')['mse'].mean()
stds=df_opt.groupby('lbda')['mse'].std()
df = pd.DataFrame(dict(mean=means, std=stds))
df['plus2']=df['mean']+2*df['std']
df['minus2']=df['mean']-2*df['std']

#minimal mean value
min_mean=min(df['mean'].values)
#print min_mean

#Find lambda where mean -2std interects minimal mean
smaller=np.where(df['minus2'].values<min_mean)[0]
while smaller[-1]>smaller[-2]+1:
  smaller=smaller[:-1]
first_smaller=smaller[-1]

y1=np.log10(df['minus2'].index[first_smaller])
y2=np.log10(df['minus2'].index[first_smaller+1])
x1=df['minus2'].iloc[first_smaller]
x2=df['minus2'].iloc[first_smaller+1]

k=(y2-y1)/(x2-x1)
m=y1-k*x1
lbda_star = 10**(k*min_mean+m)
print('lbda_start: ',lbda_star)

#n_points = df.shape[0]
#plot_mat=np.concatenate((df.index.values.reshape((n_points,1)),df['mean'].values.reshape((n_points,1)),df['plus2'].values.reshape((n_points,1)),df['minus2'].values.reshape((n_points,1))),axis=1)
#
#fig=plt.figure(figsize=(10,10))
#ax=fig.add_subplot(1,1,1)
#ax.semilogx(plot_mat[:,0],plot_mat[:,1],'C0')
#ax.semilogx(plot_mat[:,0],plot_mat[:,2],'C0--')
#ax.semilogx(plot_mat[:,0],plot_mat[:,3],'C0--')
#
#ax.axhline(min_mean,color='k')
#ax.axvline(lbda_star,color='k')
#ax.set_xlabel('$\lambda$',fontsize=FS)
#ax.set_ylabel('MSE',fontsize=FS)
#ax.text(lbda_star,-0.003,'$\lambda^*$')
#ax.text(1e-5,min_mean,'min(MSE)')
#ax.set_ylim([0,0.04])
#plt.savefig('figures/select_lbda.png')
#
