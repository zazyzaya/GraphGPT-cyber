## declare an array variable
#declare -a pcts=("0.25" "0.1" "0.5" "0.75" "0.05" "0.01" "0.005" "0.001")
declare -a pcts=("0.25")

## loop through above array
for i in "${pcts[@]}"
do
   mkdir pretrained/rw_sampling/optc_e8/"$i"_tr
   python rw_pretrain.py --optc --device $1 --tr-size $i --log-out pretrained/rw_sampling/optc_e8/"$i"_tr
   python ftbert_finetune.py --optc --device $1 --walk-len $2 --model-fname pretrained/rw_sampling/optc_e8/"$i"_tr/rw_bert_optc_tiny-best.pt --tag "$i"-base
done