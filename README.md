This is the code used in the article **Flexible, Non-parametric Modeling Using Regularized Neural Networks**, available at https://arxiv.org/abs/2012.11369 and https://link.springer.com/content/pdf/10.1007%2Fs00180-021-01190-4.pdf.
All code is run using Python 2.7.15 and TensorFlow 1.10.1.

## Figures and Tables in Section 3.1:
```
$ cd legendre
```
### Figure 2a:
```
$ Rscript make_graph.R
```
### Figure 2b:
```
$ python plot_legendre.py
```
### Figure 2c
```
$ bash lbda_sweep.sh
$ python plot_lbda_sweep.py #with correct log file on line 7
```

### Table 1
```
$ bash var_imp.sh
$ python var_imp_tab.py #with correct log file on lines 5, 6
```
### Table 2
```
$ bash fcts.sh
$ python fcts_tab.py #with correct log file on line 5
```

## Figures and Tables in Section 3.3:
```
$ cd black_smoke
```
### Table 4:
```
$ bash lbda_sweep.sh #with correct model on line 33 in lbda_sweep.py
$ python plot_lbda_sweep.py #with correct log file on line 6

$ bash fcts.sh #with correct model on line 33 in fcts.py
$ python fcts_tab.py #with correct log file on line 5
```

### Figure 3:
```
$ python plot_synth_bs.py #with correct model on lines 34 and 37
```
### Figure 4:
```
$ python plot_synth_bs_lasso.py #with correct model on lines 34 and 37
```

