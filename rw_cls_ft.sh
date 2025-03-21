echo size $1, device $2
python rw_lp_finetune.py --temporal --size $1 --device $2 --walk-len 0
python rw_lp_finetune.py --temporal --size $1 --device $2 --walk-len 1
python rw_lp_finetune.py --temporal --size $1 --device $2 --walk-len 2
python rw_lp_finetune.py --temporal --size $1 --device $2 --walk-len 3
python rw_lp_finetune.py --temporal --size $1 --device $2 --walk-len 4
python rw_lp_finetune.py --temporal --size $1 --device $2 --walk-len 5
python rw_lp_finetune.py --temporal --size $1 --device $2 --walk-len 6
python rw_lp_finetune.py --temporal --size $1 --device $2 --walk-len 7
python rw_lp_finetune.py --temporal --size $1 --device $2 --walk-len 8
python rw_lp_finetune.py --temporal --size $1 --device $2 --walk-len 9
python rw_lp_finetune.py --temporal --size $1 --device $2 --walk-len 10