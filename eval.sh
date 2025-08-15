export MUJOCO_GL="egl"
export __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
export CUDA_VISIBLE_DEVICES=0

python experiments/robot/libero/run_libero_eval.py \
    --model_family llava \
    --task_suite_name libero_3_among_90 \
    --center_crop False \
    --use_wandb True \
    --wandb_project ecot_libero \
    --wandb_entity AGI_CSI