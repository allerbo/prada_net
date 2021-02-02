import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines
W1 = np.loadtxt('data/W1.txt')
W2 = np.loadtxt('data/W2.txt')
W2 = W2.reshape((W2.shape[0],1))
b1 = np.loadtxt('data/b1.txt')
b2 = np.loadtxt('data/b2.txt')
b2 = b2.reshape((1,))
x=np.tile(np.arange(-1,1,0.01),(W1.shape[0],1)).transpose()

fig,axes=plt.subplots(4,5,figsize=(7,6))

plt.rc('xtick',labelsize=15)
plt.rc('ytick',labelsize=15)
for i in range(4):
  selector = [c for c in range(x.shape[1]) if c != i]
  xi=x.copy()
  xi[:,selector]=0
  xi_v = xi[:,i]
  yi = np.matmul(np.tanh(np.matmul(xi, W1)+ b1), W2) + b2
  axes[i,0].plot(xi_v,yi,color='C'+str(i))
  if i==1:
    nl='\n'
  else:
    nl=''
  axes[i,0].set_ylabel('$\^P_'+str(i+1)+'$(x)'+nl)
  
  
  j=0
  for idx in np.where(np.abs(W1[i,:])>0)[0]:
    W1_1 = W1.copy()
    selector = [c for c in range(W1_1.shape[1]) if c != idx]
    W1_1[:,selector]=0
    y_idx = np.matmul(np.tanh(np.matmul(xi, W1_1)+ b1), W2) + b2
    j+=1
    axes[i,j].plot(xi_v,y_idx,color='C'+str(i))
  for ii in range(4-j):
    fig.delaxes(axes[i,4-ii])


l1 = matplotlib.lines.Line2D([120,120], [0, 1000],color="#000000", linewidth=3)
fig.lines.extend([l1])
axes[0,0].set_title(' ')
plt.gcf().text(0.01, .95, 'Entire Polynomial', fontsize=12)
plt.gcf().text(.28, .95, 'Subfunctions', fontsize=12)
fig.tight_layout()
plt.savefig('figures/plot_legendre.pdf')
