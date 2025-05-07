echo size $1, device $2
python snapshot_finetune.py --size $1 --device $2 --walk-len 1 --unsw
python snapshot_finetune.py --size $1 --device $2 --walk-len 2 --unsw
python snapshot_finetune.py --size $1 --device $2 --walk-len 3 --unsw
python snapshot_finetune.py --size $1 --device $2 --walk-len 4 --unsw
python snapshot_finetune.py --size $1 --device $2 --walk-len 5 --unsw
python snapshot_finetune.py --size $1 --device $2 --walk-len 16 --unsw
python snapshot_finetune.py --size $1 --device $2 --walk-len 32 --unsw