echo size $1, device $2, static $3
python snapshot_finetune.py --size $1 --device $2 --walk-len 0 --unsw $3
python snapshot_finetune.py --size $1 --device $2 --walk-len 1 --unsw $3
python snapshot_finetune.py --size $1 --device $2 --walk-len 2 --unsw $3
python snapshot_finetune.py --size $1 --device $2 --walk-len 4 --unsw $3
python snapshot_finetune.py --size $1 --device $2 --walk-len 6 --unsw $3
python snapshot_finetune.py --size $1 --device $2 --walk-len 8 --unsw $3
python snapshot_finetune.py --size $1 --device $2 --walk-len 10 --unsw $3
python snapshot_finetune.py --size $1 --device $2 --walk-len 16 --unsw $3
python snapshot_finetune.py --size $1 --device $2 --walk-len 32 --unsw $3
python snapshot_finetune.py --size $1 --device $2 --walk-len 0 --unsw --bi $3
python snapshot_finetune.py --size $1 --device $2 --walk-len 1 --unsw --bi $3
python snapshot_finetune.py --size $1 --device $2 --walk-len 2 --unsw --bi $3
python snapshot_finetune.py --size $1 --device $2 --walk-len 4 --unsw --bi $3
python snapshot_finetune.py --size $1 --device $2 --walk-len 6 --unsw --bi $3
python snapshot_finetune.py --size $1 --device $2 --walk-len 8 --unsw --bi $3
python snapshot_finetune.py --size $1 --device $2 --walk-len 10 --unsw --bi $3
python snapshot_finetune.py --size $1 --device $2 --walk-len 16 --unsw --bi $3
python snapshot_finetune.py --size $1 --device $2 --walk-len 32 --unsw --bi $3