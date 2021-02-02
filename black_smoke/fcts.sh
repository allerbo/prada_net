date1=`date`
date2="${date1:11:8}"
log_name="fcts_${date2//:/_}"

for noise_seed in {1..50}
do
  for opt_seed in {1..5}
  do
    python fcts.py $log_name $noise_seed $opt_seed
  done
done
