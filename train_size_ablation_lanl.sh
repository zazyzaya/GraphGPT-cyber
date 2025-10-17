## declare an array variable
declare -a pcts=("0.1" "0.5" "0.75")

## loop through above array
for i in "${pcts[@]}"
do
   mkdir pretrained/rw_sampling/lanl14argus/"$i"_tr
   python rw_pretrain.py --argus --device $1 --tr-size $i --log-out pretrained/rw_sampling/lanl14argus/"$i"_tr
   python snapshot_finetune.py --argus --device $1 --tr-size $i --walk-len 1 --model-fname pretrained/rw_sampling/lanl14argus/"$i"_tr/rw_bert_lanl14argus_tiny.pt --static
done