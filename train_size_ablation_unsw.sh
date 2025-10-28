## declare an array variable
#declare -a pcts=("0.25" "0.1" "0.5" "0.75" "0.05" "0.01" "0.005" "0.001")
declare -a pcts=("0.25")

## loop through above array
for i in "${pcts[@]}"
do
   mkdir pretrained/snapshot_rw/unsw_e8/"$i"_tr
   python rw_pretrain.py --unsw --device $1 --tr-size $i --trw --log-out pretrained/snapshot_rw/unsw_e8/"$i"_tr
   python snapshot_finetune.py --unsw --device $1 --walk-len $2 --model-fname pretrained/rw_sampling/unsw_e8/"$i"_tr/trw_bert_unsw_tiny-best.pt --tag "$i"-base
done