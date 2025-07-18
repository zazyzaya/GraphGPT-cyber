#python snapshot_finetune.py --device $1 --walk-len 0 --lanlcomp --static
#python snapshot_finetune.py --device $1 --walk-len 0 --lanlcomp --bi --static
#python snapshot_finetune.py --device $1 --walk-len 1 --lanlcomp --static
python snapshot_finetune.py --device $1 --walk-len 1 --lanlcomp --bi --static
#python snapshot_finetune.py --device $1 --walk-len 2 --lanlcomp --static
python snapshot_finetune.py --device $1 --walk-len 2 --lanlcomp --bi --static
python snapshot_finetune.py --device $1 --walk-len 4 --lanlcomp --static
python snapshot_finetune.py --device $1 --walk-len 4 --lanlcomp --bi --static
python snapshot_finetune.py --device $1 --walk-len 6 --lanlcomp --static
python snapshot_finetune.py --device $1 --walk-len 6 --lanlcomp --bi --static
python snapshot_finetune.py --device $1 --walk-len 8 --lanlcomp --static
python snapshot_finetune.py --device $1 --walk-len 8 --lanlcomp --bi --static
python snapshot_finetune.py --device $1 --walk-len 10 --lanlcomp --static
python snapshot_finetune.py --device $1 --walk-len 10 --lanlcomp --bi --static
python snapshot_finetune.py --device $1 --walk-len 16 --lanlcomp --static
python snapshot_finetune.py --device $1 --walk-len 16 --lanlcomp --bi --static
python snapshot_finetune.py --device $1 --walk-len 32 --lanlcomp --static
python snapshot_finetune.py --device $1 --walk-len 32 --lanlcomp --bi --static