echo "Searching for the GPU with maximum free memory..."

gpu_id=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | \
         awk 'BEGIN{max=0; id=0} {if($1>max){max=$1; id=NR-1}} END{print id}')

export CUDA_VISIBLE_DEVICES=$gpu_id

echo "Selected GPU: $CUDA_VISIBLE_DEVICES"

python run_inference.py
