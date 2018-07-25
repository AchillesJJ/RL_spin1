#!bin/bash

export core_num=6

# for ((cnt = 1; cnt <= $core_num; cnt++))
# do
#   export delta_time=`echo "scale=6; 0.05*$cnt" | bc`
#   export N_realise=$cnt
#   echo $delta_time, $N_realise
#   nohup python main.py $delta_time $N_realise > pso.file 2>&1 &
# done

for ((cnt = 1; cnt <= $core_num; cnt++))
do
  export trial_num=$(($cnt+12))
  echo $trial_num
  nohup python main.py $trial_num > pso.file 2>&1 &
done