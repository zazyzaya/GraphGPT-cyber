#python snapshot_finetune.py --device $1 --argus --walk-len 4 --static --from-random 
python snapshot_finetune.py --device $1 --unsw --walk-len 8 --static --from-random 
python ftbert_finetune.py --device $1 --optc --walk-len 1 --static --from-random 
python snapshot_finetune.py --device $1 --argus --walk-len 4 --from-random 
python snapshot_finetune.py --device $1 --unsw --walk-len 8 --from-random 
python ftbert_finetune.py --device $1 --optc --walk-len 6 --from-random 