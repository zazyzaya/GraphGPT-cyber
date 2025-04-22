echo size $1, device $2
python snapshot_finetune.py --size $1 --device $2 --walk-len 0 --optc
python snapshot_finetune.py --size $1 --device $2 --walk-len 2 --optc
python snapshot_finetune.py --size $1 --device $2 --walk-len 4 --optc
python snapshot_finetune.py --size $1 --device $2 --walk-len 6 --optc
python snapshot_finetune.py --size $1 --device $2 --walk-len 8 --optc
python snapshot_finetune.py --size $1 --device $2 --walk-len 10 --optc