# Convert ModernGena to HF:

To export model to HF, run something like following 

```bash
python convert_to_hf.py --output-name hf --output-dir runs/moderngena-base-pretrain-promoters_multi_v2_resume_ep30-ba90700/hf/ --input-checkpoint runs/moderngena-base-pretrain-promoters_multi_v2_resume_ep30-ba90700/latest-rank0.pt --cls-token-id 1 --sep-token-id 2 --pad-token-id 3 --mask-token-id 4 --max-length 1024
```