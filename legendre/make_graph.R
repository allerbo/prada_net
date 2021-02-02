links = function(mat, from_to){
  from = paste0(from_to[1], '_')
  to=ifelse(dim(mat)[2]>1,paste0(from_to[2], '_'),from_to[2])
  out_str = ''
  for (r in 1:nrow(mat)) {
    for (c in 1:ncol(mat)) {
      if(mat[r,c] == 1) {
        out_str = paste0(out_str, '  ',from,r,' -> ', to,ifelse(dim(mat)[2]>1,c,''),';\n')
      }
 
    }
  }
  return(out_str)
}


cluster = function(nodes, label,no_lbl=F) {
  coll = ' '
  if (no_lbl) {
    lbl = ', label=""'
  } else {
    lbl = ''
  }

  if (no_lbl) {
    paste0('  subgraph cluster_',label,' {\n    color=white;\n    node [style=solid,color=black, shape=circle, label=""];\n    ', ifelse(label=='y', label,paste0(label,'_',nodes,collapse=coll)),';\n    ','label= "";\n  }\n')
  } else {
    paste0('  subgraph cluster_',label,' {\n    color=white;\n    node [style=solid,color=black, shape=circle];\n    ', ifelse(label=='y', label,paste0(label,'_',nodes,'[label=<',label,'<SUB>',nodes,'</SUB>>]',collapse=coll)),';\n    ','label= "";\n  }\n')
  }
}

lbl = ''

W1 = as.matrix(read.csv(paste0('data/W1.txt'), sep=' ',header=F) !=0)
W2 = as.matrix(read.csv(paste0('data/W2.txt'), sep=' ',header=F) !=0)

mode(W1) = 'numeric'
mode(W2) = 'numeric'
W1[is.na(W1)] = 0
W2[is.na(W2)] = 0


len_h1 = dim(W1)[2]
unused_h1 = c()
for (rc in 1:len_h1) {
  if (sum(W1[,rc])==0 && sum(W2[rc,])==0) {
    W1[,rc] = rep(0,dim(W1)[1])
    W2[rc,] = rep(0,dim(W2)[2])
    unused_h1 = c(unused_h1,rc)
  }
}

init_str = paste0('digraph G {\n  rankdir=LR\n  splines=line\n  nodesep=0.05;\n  ranksep=1;\n',lbl)
x_nodes = 1:dim(W1)[1]
h1_nodes = 1:dim(W1)[2]
y_nodes = 1:dim(W2)[2]
h1_nodes = h1_nodes[!(h1_nodes %in% unused_h1)] #remove empty ls
cluster_x = cluster(x_nodes,'x',no_lbl=F)
cluster_h1 = cluster(h1_nodes,'h1',no_lbl=T)
cluster_y = cluster(y_nodes,'y',no_lbl=F)
layer_x_h1 = links(W1, c('x','h1'))
layer_h1_y = links(W2, c('h1','y'))
out_str_W = paste0(init_str, cluster_x, cluster_h1, cluster_y, layer_x_h1, layer_h1_y, '}')


write(out_str_W, 'figures/graph')
system('dot -Tpdf -O figures/graph')


