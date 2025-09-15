#python snapshot_finetune.py --device $1 --walk-len 3 --argus --static 
python snapshot_finetune.py --device $1 --walk-len 5 --argus --static
python snapshot_finetune.py --device $1 --walk-len 7 --argus --static
python snapshot_finetune.py --device $1 --walk-len 9 --argus --static
python snapshot_finetune.py --device $1 --walk-len 10 --argus --static
python snapshot_finetune.py --device $1 --walk-len 16 --argus --static
python snapshot_finetune.py --device $1 --walk-len 32 --argus --static