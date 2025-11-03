mkdir pretrained/rw_sampling/epoch_ablation/

## declare an array variable
declare -a pcts=("1" "2" "3" "5" "7" "9" "10")

## loop through above array
#for i in "${pcts[@]}"
#do
#    mkdir -p pretrained/rw_sampling/epoch_ablation_2048/"$i"e8/
#    mkdir -p results/rw/lanl14argus/epoch_ablation_2048/"$i"e8/
#done 

(
    python rw_pretrain.py --argus --n-tokens 10 --log-out pretrained/rw_sampling/epoch_ablation_2048/10e8/ --device 0;
    ./lanl_ft.sh 0 "--static --special --model-fname pretrained/rw_sampling/epoch_ablation_2048/10e8/rw_bert_lanl14argus_tiny.pt --out-dir results/rw/lanl14argus/epoch_ablation_2048/10e8"
    ./lanl_ft.sh 0 "--static --special --model-fname pretrained/rw_sampling/epoch_ablation/10e8/rw_bert_lanl14argus_tiny.pt --out-dir results/rw/lanl14argus/epoch_ablation/10e8"
) & 
(
    python rw_pretrain.py --argus --n-tokens 1 --log-out pretrained/rw_sampling/epoch_ablation_2048/1e8/ --device 1;
    python rw_pretrain.py --argus --n-tokens 3 --log-out pretrained/rw_sampling/epoch_ablation_2048/3e8/ --device 1; 
    python rw_pretrain.py --argus --n-tokens 5 --log-out pretrained/rw_sampling/epoch_ablation_2048/5e8/ --device 1;
    ./lanl_ft.sh 1 "--static --special --model-fname pretrained/rw_sampling/epoch_ablation_2048/3e8/rw_bert_lanl14argus_tiny.pt --out-dir results/rw/lanl14argus/epoch_ablation_2048/3e8"
    ./lanl_ft.sh 1 "--static --special --model-fname pretrained/rw_sampling/epoch_ablation/3e8/rw_bert_lanl14argus_tiny.pt --out-dir results/rw/lanl14argus/epoch_ablation/3e8"
    ./lanl_ft.sh 1 "--static --special --model-fname pretrained/rw_sampling/epoch_ablation_2048/5e8/rw_bert_lanl14argus_tiny.pt --out-dir results/rw/lanl14argus/epoch_ablation_2048/5e8"
    ./lanl_ft.sh 1 "--static --special --model-fname pretrained/rw_sampling/epoch_ablation/5e8/rw_bert_lanl14argus_tiny.pt --out-dir results/rw/lanl14argus/epoch_ablation/5e8"
) &
(
    python rw_pretrain.py --argus --n-tokens 2 --log-out pretrained/rw_sampling/epoch_ablation_2048/2e8/ --device 2; 
    python rw_pretrain.py --argus --n-tokens 7 --log-out pretrained/rw_sampling/epoch_ablation_2048/7e8/ --device 2;
    ./lanl_ft.sh 2 "--static --special --model-fname pretrained/rw_sampling/epoch_ablation_2048/2e8/rw_bert_lanl14argus_tiny.pt --out-dir results/rw/lanl14argus/epoch_ablation_2048/2e8";
    ./lanl_ft.sh 2 "--static --special --model-fname pretrained/rw_sampling/epoch_ablation/2e8/rw_bert_lanl14argus_tiny.pt --out-dir results/rw/lanl14argus/epoch_ablation/2e8";
    ./lanl_ft.sh 2 "--static --special --model-fname pretrained/rw_sampling/epoch_ablation_2048/7e8/rw_bert_lanl14argus_tiny.pt --out-dir results/rw/lanl14argus/epoch_ablation_2048/7e8"
    ./lanl_ft.sh 2 "--static --special --model-fname pretrained/rw_sampling/epoch_ablation/7e8/rw_bert_lanl14argus_tiny.pt --out-dir results/rw/lanl14argus/epoch_ablation/7e8"
) &
( 
    python rw_pretrain.py --argus --n-tokens 9 --log-out pretrained/rw_sampling/epoch_ablation_2048/9e8/ --device 3;
    ./lanl_ft.sh 3 "--static --special --model-fname pretrained/rw_sampling/epoch_ablation_2048/9e8/rw_bert_lanl14argus_tiny.pt --out-dir results/rw/lanl14argus/epoch_ablation_2048/9e8"
    ./lanl_ft.sh 3 "--static --special --model-fname pretrained/rw_sampling/epoch_ablation/9e8/rw_bert_lanl14argus_tiny.pt --out-dir results/rw/lanl14argus/epoch_ablation/9e8"
    ./lanl_ft.sh 3 "--static --special --model-fname pretrained/rw_sampling/epoch_ablation/1e8/rw_bert_lanl14argus_tiny.pt --out-dir results/rw/lanl14argus/epoch_ablation/1e8";
    ./lanl_ft.sh 3 "--static --special --model-fname pretrained/rw_sampling/epoch_ablation_2048/1e8/rw_bert_lanl14argus_tiny.pt --out-dir results/rw/lanl14argus/epoch_ablation_2048/1e8";
)