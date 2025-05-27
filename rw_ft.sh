echo size $1, device $2, dataset $3
python rw_finetune.py --size $1 --device $2 --walk-len 1 $3
python rw_finetune.py --size $1 --device $2 --walk-len 2 $3
python rw_finetune.py --size $1 --device $2 --walk-len 3 $3
python rw_finetune.py --size $1 --device $2 --walk-len 4 $3
python rw_finetune.py --size $1 --device $2 --walk-len 5 $3
python rw_finetune.py --size $1 --device $2 --walk-len 16 $3
python rw_finetune.py --size $1 --device $2 --walk-len 32 $3