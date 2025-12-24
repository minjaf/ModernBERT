cfg=$1

save_dir=$(grep "run_name" ${cfg} | head -1 | cut -f2 -d " ")
save_dir="runs/${save_dir}"

# run one training step to initialize the model
CUDA_VISIBLE_DEVICES=0 composer main.py ${cfg}

# convert to hf
python convert_to_hf.py --output-name hf --output-dir ${save_dir} --input-checkpoint ${save_dir}/latest-rank0.pt --cls-token-id 1 --sep-token-id 2 --pad-token-id 3 --mask-token-id 4 --max-length 1024

echo "Model exported to ${save_dir}/hf"