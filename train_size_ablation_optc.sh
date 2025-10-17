## declare an array variable
#declare -a pcts=("0.25" "0.1" "0.5" "0.75" "0.05" "0.01" "0.005" "0.001")
declare -a pcts=("0.9")

## loop through above array
for i in "${pcts[@]}"
do
   python rw_pretrain.py --optc --device $1 --tr-size $i
   python ftbert_finetune.py --optc --device $1 --tr-size $i --walk-len $2 --model-fname rw_bert_optc_tiny-best.pt
done