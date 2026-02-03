# Standard training

## Download data

```bash
cd ~/DNALM/ModernBERT/data$  
~/.local/bin/aws s3 cp s3://genalm/data/pretraining/promoters/ . --endpoint-url https://s3.cloud.ru --recursive
```

## run (large):

```
composer main.py yamls/moderngena/gena-large-pretrain_multi_promoters_v2.yaml
```

# Prepare to tain MLM with
