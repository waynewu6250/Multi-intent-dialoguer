# Multi-intent-dialoguer
Research project for task-oriented dialogue system with jointly training multi-intent classification and slot filling

## Using BERT-as-service
Enter the following command:
>
    /mnt/2TB-NVMe/home/twu367/anaconda3/envs/bert/bin/bert-serving-start -model_dir /tmp/multi_cased_L-12_H-768_A-12 -  
    num_worker=1 -max_seq_len=NONE -max_batch_size=28
