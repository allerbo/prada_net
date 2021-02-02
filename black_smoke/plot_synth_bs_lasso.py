#0     year is year
#1     yday is day of year (Jan 1st is 1)
#2     wday is a numeric code for day of week.
#3,4   Tmin0/Tmax0 are minimum and maximum temperature in Celcius for the day in question.
#5,6   Tmean1/Tmean2 are mean temerature on the preceding 2 days (Celcius).
#7     rainfall is monthly rainfall in mm.
#8,9   x and y are easting and northing on a kilometre grid.
#10    h is elevation in metres.
#11-   type1 is a simplified classification of station types.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import sys
import itertools as it
def powerset(s):
  x = len(s)
  outs=[]
  for i in range(2** x):
    outs.append([s[j] for j in range(x) if (i & (2**j))])
  return outs[1:]

def un_bias(a,b=None):
  if b is None: 
    return a-np.mean(a)
  return a-np.mean(b)

ii_rows=4
ii_cols=4
fig,axes=plt.subplots(ii_rows,ii_cols,figsize=(2*ii_cols,2*ii_rows))
ii=0
x_types=['y','doy','dow','T^0','T^1','\\overline{T1}', '\\overline{T2}','r','e','n','h','\\alpha']


SYN_APDX='fcts_08_30_30'

#True fct
TRUE_APDX = 'bs_07_49_16_21'

TOT=True
for my_fcts_TOT in [['0','1','9','8'],['0','1','9','8-9', '9-8','8-9-10']]:
  for my_fct1 in my_fcts_TOT:
    #For each true function, identify which nodes are used to realize it
    fct_splits=my_fct1.split('-')
    my_fct='-'.join(map(str,sorted(map(int,fct_splits)))) #sort to work for fct_nodes
    fct1=int(fct_splits[0])
    if len(fct_splits)>=2:
      fct2=int(fct_splits[1])
      bins2 = np.array([-1.,  1.])
    else:
      fct2=None
      bins2 = np.array([ 0.])
    if len(fct_splits)>=3:
      fct3=int(fct_splits[2])
      bins3 = np.array([-1., 1.])
    else:
      fct3=None
      bins3 = np.array([ 0.])
    x1=np.linspace(-1.5,1.5,101)
    for x2_value in bins2:
      for x3_value in bins3:
        x_mat=np.zeros((len(x1),11))
        x_mat[:,fct1]=x1
        if not fct2 is None:
          x2 = np.array([x2_value for x1_value in x1])
          x_mat[:,fct2]=x2
        if not fct3 is None:
          x3 = np.array([x3_value for x1_value in x1])
          x_mat[:,fct3]=x3

        y_trues = []
        W1_true = np.loadtxt('data/W1_'+TRUE_APDX+'.txt')
        W2_true = np.loadtxt('data/W2_'+TRUE_APDX+'.txt')
        b1_true = np.loadtxt('data/b1_'+TRUE_APDX+'.txt')
        b2_true = np.loadtxt('data/b2_'+TRUE_APDX+'.txt')
        W2_true = W2_true.reshape((W2_true.shape[0],1))
        b2_true = b2_true.reshape((1,))
         
        fct_nodes={}
        nodes = np.where(np.sum(np.abs(W1_true),0)>0)[0]
        for node in nodes: 
          if TOT:
            fct_orig = np.where(np.abs(W1_true[:,node])>0)[0].tolist()
            for fct in powerset(fct_orig): #TOT
              fct_str='-'.join(map(str,fct))
              if not fct_str in fct_nodes.keys():
                fct_nodes[fct_str] = [node]
              else:
                fct_nodes[fct_str].append(node)
          else:
              fct = np.where(np.abs(W1_true[:,node])>0)[0].tolist()
              fct_str='-'.join(map(str,fct))
              if not fct_str in fct_nodes.keys():
                fct_nodes[fct_str] = [node]
              else:
                fct_nodes[fct_str].append(node)
        
        #Set all links, apart from the ones used in the current function, to zero and realize current function
        if my_fct in fct_nodes.keys():
          nodes = fct_nodes[my_fct]
          W1_true_sub = np.copy(W1_true)
          delete_nodes = list(range(W1_true_sub.shape[1]))
          for node in nodes:
            delete_nodes.remove(node)   
           
          W1_true_sub[:,delete_nodes]=0
          if len(W1_true_sub.shape)==1:
            W1_true_sub = W1_true_sub.reshape((1,W1_true_sub.shape[0]))
          
          b1_true_sub = np.copy(b1_true)
          y_trues.append(np.matmul(np.tanh(np.matmul(x_mat, W1_true_sub)+ b1_true_sub), W2_true) + b2_true)
        
        y_true = y_trues[0]
        y_synths=[]
        for noise_seed in range(1,51):
          W1_syn = np.loadtxt('data/W1_phase2_'+SYN_APDX+'_'+str(noise_seed)+'.txt')
          W2_syn = np.loadtxt('data/W2_phase2_'+SYN_APDX+'_'+str(noise_seed)+'.txt')
          b1_syn = np.loadtxt('data/b1_phase2_'+SYN_APDX+'_'+str(noise_seed)+'.txt')
          b2_syn = np.loadtxt('data/b2_phase2_'+SYN_APDX+'_'+str(noise_seed)+'.txt')
          W2_syn = W2_syn.reshape((W2_syn.shape[0],1))
          b2_syn = b2_syn.reshape((1,))
           
          #For each identified function, identify which nodes are used to realize it
          fct_nodes={}
          nodes = np.where(np.sum(np.abs(W1_syn),0)>0)[0]
          for node in nodes: 
            if TOT:
              fct_orig = np.where(np.abs(W1_syn[:,node])>0)[0].tolist() #TOT
              for fct in powerset(fct_orig): #TOT
                fct_str='-'.join(map(str,fct))
                if not fct_str in fct_nodes.keys():
                  fct_nodes[fct_str] = [node]
                else:
                  fct_nodes[fct_str].append(node)
            else:
              fct = np.where(np.abs(W1_syn[:,node])>0)[0].tolist()
              fct_str='-'.join(map(str,fct))
              if not fct_str in fct_nodes.keys():
                fct_nodes[fct_str] = [node]
              else:
                fct_nodes[fct_str].append(node)
           
          #Set all links, apart from the ones used in the current function, to zero and realize current function
          if my_fct in fct_nodes.keys():
            nodes = fct_nodes[my_fct]
            W1_syn_sub = np.copy(W1_syn)
            delete_nodes = list(range(W1_syn_sub.shape[1]))
            for node in nodes:
              delete_nodes.remove(node)   
            W1_syn_sub[:,delete_nodes]=0
            if len(W1_syn_sub.shape)==1:
              W1_syn_sub = W1_syn_sub.reshape((1,W1_syn_sub.shape[0]))
            b1_syn_sub = np.copy(b1_syn)
            b1_syn_sub[delete_nodes]=0
            y_synths.append(un_bias(np.matmul(np.tanh(np.matmul(x_mat, W1_syn_sub)+ b1_syn_sub), W2_syn) + b2_syn))
         
        y_synths=np.squeeze(np.array(y_synths))
         
        if len(y_synths.shape)==1:
          y_synths=(y_synths.reshape((-1,1))).T
         
        print(my_fct1,y_synths.shape)
        y_mean=np.mean(y_synths,0)
        y_std=np.std(y_synths,0)
        
        ax=axes[ii/ii_cols, ii % ii_cols]
        if y_synths.shape[1]>0:
          #Plot mean +-2std
          ax.plot(x1,y_mean,'C0')
          ax.plot(x1,y_mean+2*y_std,'C0--')
          ax.plot(x1,y_mean-2*y_std,'C0--')
        
        #Plot true
        ax.plot(x1,un_bias(y_true),'C3:')
        if TOT:
          title_text = 'f'+r'$_{tot}$'+'($'+x_types[int(fct1)]+'$'
        else:
          title_text = 'f($'+x_types[int(fct1)]+'$'
        if not fct2 is None:
          title_text+='|$'+x_types[fct2]+'$='+str(x2_value)
        if not fct3 is None:
          title_text+=',$'+x_types[fct3]+'$='+str(x3_value)
        ax.set_title(title_text+')')
        ii+=1
    if my_fct1=='9' and TOT==False:
      ii+=1
  TOT=False
  
  
fig.delaxes(axes[1,3])
fig.tight_layout()

# Get the bounding boxes of the axes including text decorations
r = fig.canvas.get_renderer()
get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
bboxes = np.array(list(map(get_bbox, axes.flat)), mtrans.Bbox).reshape(axes.shape)

#Get the minimum and maximum extent, get the coordinate half-way between those
ymax = np.array(list(map(lambda b: b.y1, bboxes.flat))).reshape(axes.shape).max(axis=1)
ymin = np.array(list(map(lambda b: b.y0, bboxes.flat))).reshape(axes.shape).min(axis=1)
ys = np.c_[ymax[1:], ymin[:-1]].mean(axis=1)
line = plt.Line2D([0,1],[ys[0],ys[0]], transform=fig.transFigure, color="black")
fig.lines.append(line)


plt.savefig('figures/bs_fcts_lasso.pdf')
