echo size $1, device $2
python rw_finetune.py --temporal --size $1 --device $2 --walk-len 0
python rw_finetune.py --temporal --size $1 --device $2 --walk-len 2
python rw_finetune.py --temporal --size $1 --device $2 --walk-len 4
python rw_finetune.py --temporal --size $1 --device $2 --walk-len 8
python rw_finetune.py --temporal --size $1 --device $2 --walk-len 10