# optc
python snapshot_finetune.py --size $1 --device $2 --walk-len 0 --optc
python snapshot_finetune.py --size $1 --device $2 --walk-len 1 --optc
python snapshot_finetune.py --size $1 --device $2 --walk-len 2 --optc
python snapshot_finetune.py --size $1 --device $2 --walk-len 4 --optc
python snapshot_finetune.py --size $1 --device $2 --walk-len 6 --optc
python snapshot_finetune.py --size $1 --device $2 --walk-len 8 --optc
python snapshot_finetune.py --size $1 --device $2 --walk-len 10 --optc
python snapshot_finetune.py --size $1 --device $2 --walk-len 16 --optc
python snapshot_finetune.py --size $1 --device $2 --walk-len 32 --optc
python snapshot_finetune.py --size $1 --device $2 --walk-len 0 --optc --bi
python snapshot_finetune.py --size $1 --device $2 --walk-len 1 --optc --bi
python snapshot_finetune.py --size $1 --device $2 --walk-len 2 --optc --bi
python snapshot_finetune.py --size $1 --device $2 --walk-len 4 --optc --bi
python snapshot_finetune.py --size $1 --device $2 --walk-len 6 --optc --bi
python snapshot_finetune.py --size $1 --device $2 --walk-len 8 --optc --bi
python snapshot_finetune.py --size $1 --device $2 --walk-len 10 --optc --bi
python snapshot_finetune.py --size $1 --device $2 --walk-len 16 --optc --bi
python snapshot_finetune.py --size $1 --device $2 --walk-len 32 --optc --bi
python snapshot_finetune.py --size $1 --device $2 --walk-len 0 --optc --static
python snapshot_finetune.py --size $1 --device $2 --walk-len 1 --optc --static
python snapshot_finetune.py --size $1 --device $2 --walk-len 2 --optc --static
python snapshot_finetune.py --size $1 --device $2 --walk-len 4 --optc --static
python snapshot_finetune.py --size $1 --device $2 --walk-len 6 --optc --static
python snapshot_finetune.py --size $1 --device $2 --walk-len 8 --optc --static
python snapshot_finetune.py --size $1 --device $2 --walk-len 10 --optc --static
python snapshot_finetune.py --size $1 --device $2 --walk-len 16 --optc --static
python snapshot_finetune.py --size $1 --device $2 --walk-len 32 --optc --static
python snapshot_finetune.py --size $1 --device $2 --walk-len 0 --optc --bi --static
python snapshot_finetune.py --size $1 --device $2 --walk-len 1 --optc --bi --static
python snapshot_finetune.py --size $1 --device $2 --walk-len 2 --optc --bi --static
python snapshot_finetune.py --size $1 --device $2 --walk-len 4 --optc --bi --static
python snapshot_finetune.py --size $1 --device $2 --walk-len 6 --optc --bi --static
python snapshot_finetune.py --size $1 --device $2 --walk-len 8 --optc --bi --static
python snapshot_finetune.py --size $1 --device $2 --walk-len 10 --optc --bi --static
python snapshot_finetune.py --size $1 --device $2 --walk-len 16 --optc --bi --static
python snapshot_finetune.py --size $1 --device $2 --walk-len 32 --optc --bi --static

# UNSW is bad with the FTBert model bc it's hard to make negative edge samples
# when edges have features. Just do a short sweep to collect data showing this
python ftbert_finetune.py --walk-len 4 --size tiny --device 2 --unsw
python ftbert_finetune.py --walk-len 4 --size tiny --device 2 --unsw --bi 
python ftbert_finetune.py --walk-len 4 --size tiny --device 2 --unsw --static
python ftbert_finetune.py --walk-len 4 --size tiny --device 2 --unsw --bi --static
python ftbert_finetune.py --walk-len 8 --size tiny --device 2 --unsw
python ftbert_finetune.py --walk-len 8 --size tiny --device 2 --unsw --bi 
python ftbert_finetune.py --walk-len 8 --size tiny --device 2 --unsw --static
python ftbert_finetune.py --walk-len 8 --size tiny --device 2 --unsw --bi --static
python ftbert_finetune.py --walk-len 16 --size tiny --device 2 --unsw
python ftbert_finetune.py --walk-len 16 --size tiny --device 2 --unsw --bi 
python ftbert_finetune.py --walk-len 16 --size tiny --device 2 --unsw --static
python ftbert_finetune.py --walk-len 16 --size tiny --device 2 --unsw --bi --static
python ftbert_finetune.py --walk-len 32 --size tiny --device 2 --unsw
python ftbert_finetune.py --walk-len 32 --size tiny --device 2 --unsw --bi 
python ftbert_finetune.py --walk-len 32 --size tiny --device 2 --unsw --static
python ftbert_finetune.py --walk-len 32 --size tiny --device 2 --unsw --bi --static