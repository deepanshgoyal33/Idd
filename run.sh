WEIGHT_DIRECTORY="weight"

if [ ! -d "$WEIGHT_DIRECTORY" ]; then
  mkdir "$WEIGHT_DIRECTORY"
  wget 
fi

CUDA_VISIBLE_DEVICES=0 python train.py
