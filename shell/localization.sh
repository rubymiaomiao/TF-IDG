echo "Searching for the GPU with maximum free memory..."
gpu_id=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | \
         awk 'BEGIN{max=0; id=0} {if($1>max){max=$1; id=NR-1}} END{print id}')

export CUDA_VISIBLE_DEVICES=$gpu_id
echo "Selected GPU: $CUDA_VISIBLE_DEVICES"


path_to_the_generated_data="path_to_the_generated_data"
path_to_mvtec="path_to_mvtec"
checkpoint_path="./anomaly_detection/checkpoints/mvtec/localization"

# python train-localization.py --generated_data_path $path_to_the_generated_data  --mvtec_path=$path_to_mvtec --save_path=$checkpoint_path \
#                              --epochs=200  --numbers=500 --sample_name="all"
python test-localization.py --mvtec_path=$path_to_mvtec --checkpoint_path=$checkpoint_path --sample_name="all"