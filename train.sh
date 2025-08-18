torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/train.py \
  --vla.type "prism-dinosiglip-224px+mx-libero-90" \
  --data_root_dir "/home/work/AGI_NIH/data/embodied_features_and_demos_libero" \
  --run_root_dir "/home/work/AGI_NIH/checkpoints/ecot_libero_2" \
  --wandb_project ecot \
  --wandb_entity AGI_CSI
