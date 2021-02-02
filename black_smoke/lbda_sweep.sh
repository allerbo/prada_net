date1=`date`
date2="${date1:11:8}"
log_name="lbda_sweep_${date2//:/_}"

for boot_seed in {1..20}
do
  for opt_seed in {1..5}
  do
    python2 lbda_sweep.py $log_name $boot_seed $opt_seed
    for i in {-8..2}
    do
      lbda=`bc -l <<< "scale=7; e($i/2*l(10))"`
      python2 lbda_sweep.py $log_name $boot_seed $opt_seed $lbda
    done
  done
done
