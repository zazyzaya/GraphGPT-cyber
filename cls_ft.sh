echo size $1, device $2, dataset $3, static $4
python ftbert_finetune.py --walk-len 0 --size $1 --device $2 $3 $4 
python ftbert_finetune.py --walk-len 1 --size $1 --device $2 $3 $4
python ftbert_finetune.py --walk-len 2 --size $1 --device $2 $3 $4
python ftbert_finetune.py --walk-len 3 --size $1 --device $2 $3 $4
python ftbert_finetune.py --walk-len 4 --size $1 --device $2 $3 $4
python ftbert_finetune.py --walk-len 5 --size $1 --device $2 $3 $4
python ftbert_finetune.py --walk-len 16 --size $1 --device $2 $3 $4
python ftbert_finetune.py --walk-len 32 --size $1 --device $2 $3 $4
#python ftbert_finetune.py --walk-len 0 --size $1 --device $2 $3 $4 --bi
#python ftbert_finetune.py --walk-len 1 --size $1 --device $2 $3 $4 --bi
#python ftbert_finetune.py --walk-len 2 --size $1 --device $2 $3 $4 --bi
#python ftbert_finetune.py --walk-len 3 --size $1 --device $2 $3 $4 --bi
#python ftbert_finetune.py --walk-len 4 --size $1 --device $2 $3 $4 --bi
#python ftbert_finetune.py --walk-len 5 --size $1 --device $2 $3 $4 --bi
#python ftbert_finetune.py --walk-len 16 --size $1 --device $2 $3 $4 --bi
#python ftbert_finetune.py --walk-len 32 --size $1 --device $2 $3 $4 --bi