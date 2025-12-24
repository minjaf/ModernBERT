# Convert ModernGena to HF:

To export model to HF, run something like following 

```bash
cpt="runs/moderngena-large-based-expression-decoder"; python convert_to_hf.py --output-name hf --output-dir ${cpt}/ --input-checkpoint ${cpt}/latest-rank0.pt --cls-token-id 1 --sep-token-id 2 --pad-token-id 3 --mask-token-id 4 --max-length 1024
```

# Create random init for model and export it to HF:

See example configs in: yamls/moderngena/expression_decoder/

```bash
bash export_random_init_model.sh /path/to/config
```