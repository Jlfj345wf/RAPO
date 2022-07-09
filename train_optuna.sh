cpu_id1=`expr $1 \* 3`
cpu_id2=`expr $1 \* 3 + 1`
cpu_id3=`expr $1 \* 3 + 2`
cpu_id4=`expr $2 \* 3`
cpu_id5=`expr $2 \* 3 + 1`
cpu_id6=`expr $2 \* 3 + 2`
echo $cpu_id1
echo $cpu_id2
echo $cpu_id3
echo $cpu_id4
echo $cpu_id5
echo $cpu_id6
CUDA_VISIBLE_DEVICES=$1,$2 taskset -c $cpu_id1,$cpu_id2,$cpu_id3,$cpu_id4,$cpu_id5,$cpu_id6 python new_optuna_train.py \
            --src_lang "es" \
            --tgt_lang "en" \
            --adapter_type "shift" \
            --n_trail 50