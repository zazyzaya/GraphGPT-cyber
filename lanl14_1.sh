#python snapshot_finetune.py --device $1 --walk-len 0 --lanlflows --static
#python snapshot_finetune.py --device $1 --walk-len 0 --lanlflows --bi --static
python snapshot_finetune.py --device $1 --walk-len 1 --lanlflows --static
python snapshot_finetune.py --device $1 --walk-len 1 --lanlflows --bi --static
python snapshot_finetune.py --device $1 --walk-len 2 --lanlflows --static
python snapshot_finetune.py --device $1 --walk-len 2 --lanlflows --bi --static
python snapshot_finetune.py --device $1 --walk-len 4 --lanlflows --static
python snapshot_finetune.py --device $1 --walk-len 4 --lanlflows --bi --static
python snapshot_finetune.py --device $1 --walk-len 6 --lanlflows --static
python snapshot_finetune.py --device $1 --walk-len 6 --lanlflows --bi --static
python snapshot_finetune.py --device $1 --walk-len 8 --lanlflows --static
python snapshot_finetune.py --device $1 --walk-len 8 --lanlflows --bi --static
python snapshot_finetune.py --device $1 --walk-len 10 --lanlflows --static
python snapshot_finetune.py --device $1 --walk-len 10 --lanlflows --bi --static
python snapshot_finetune.py --device $1 --walk-len 16 --lanlflows --static
python snapshot_finetune.py --device $1 --walk-len 16 --lanlflows --bi --static
python snapshot_finetune.py --device $1 --walk-len 32 --lanlflows --static
python snapshot_finetune.py --device $1 --walk-len 32 --lanlflows --bi --static