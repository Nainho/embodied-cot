"""Utils for evaluating policies in LIBERO simulation environments."""
import sys
import time
from functools import partial
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from accelerate.utils import set_seed
from PIL import Image

from prismatic.models import load_vla
from prismatic.models.materialize import VISION_BACKBONES
from prismatic.util.cot_utils import CotTag, get_cot_tags_list

sys.path.append("../..")  # hack so that the interpreter can find experiments.robot


import math
import os

import tensorflow as tf
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

import json

from transformers import (
    AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.vla.action_tokenizer import ActionTokenizer

ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")




def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_libero_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    #img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing  기존 상하좌우 반전 180
    img = img[::-1] # 상하반전만
    #img = img[:,::-1] # 좌우반전만
    img = resize_image(img, resize_size)
    return img


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}/{DATE_TIME}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

############################################################################################

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
BRIDGE_PROPRIO_DIM = 7
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})


def get_vla_image_resize_size(vision_backbone_id: str) -> int:
    """Gets VLA image resize size from vision backbone ID."""
    return VISION_BACKBONES[vision_backbone_id]["kwargs"]["default_image_size"]


def get_image_resize_size(cfg):
    """
    Gets image resize size for a model class.
    If `resize_size` is an int, then the resized image will be a square.
    Else, the image will be a rectangle.
    """
    if cfg.model_family == "llava":
        resize_size = 224
        #get_vla_image_resize_size(cfg.model.vision_backbone_id)
    elif cfg.model_family == "octo":
        resize_size = 256
    elif cfg.model_family == "rt_1_x":
        resize_size = (640, 480)  # PIL expects (W, H)
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return resize_size


def _is_hf_dir(path: str | Path) -> bool:
    p = Path(path)
    return p.is_dir() and (p / "model.safetensors.index.json").exists()

def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint (HF dir or .pt)."""
    print(f"[*] Initializing Generation Playground with `{cfg.model_family}`")
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    set_seed(cfg.seed)

    ckpt = Path(cfg.pretrained_checkpoint)
    print(f"Loading VLM from checkpoint: {ckpt}")
    
    # === Case A: HF 포맷 디렉터리 ===
    if _is_hf_dir(ckpt):
        print("[HF] Detected Hugging Face-format directory. Loading via AutoClasses...")

        # 1) AutoClass 등록 (허브 푸시 모델이면 생략 가능 / 로컬은 필요)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        # 2) Processor & Model 로드 (처음엔 fp32로 로드해 기존 assert와 정합)
        processor = AutoProcessor.from_pretrained(str(ckpt), trust_remote_code=True)
        vla = AutoModelForVision2Seq.from_pretrained(
            str(ckpt),
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        vla.to(DEVICE)
        vla.eval()

        # 평가 파이프라인에서 필요할 수 있는 부속을 모델에 꽂아둠
        vla.processor = processor
        ds_stats = ckpt / "dataset_statistics.json"
        vla.norm_stats = json.loads(ds_stats.read_text()) if ds_stats.exists() else None
        vla.action_tokenizer = ActionTokenizer(processor.tokenizer)

        return vla

    # === Case B: 기존 .pt 체크포인트 경로 ===
    else:
        vla = load_vla(str(ckpt), hf_token=hf_token, load_for_training=False)
        for param in vla.parameters():
            assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"
        # Cast to half precision.
        vla.vision_backbone.to(dtype=vla.vision_backbone.half_precision_dtype)
        vla.llm_backbone.to(dtype=vla.llm_backbone.half_precision_dtype)
        vla.to(dtype=vla.llm_backbone.half_precision_dtype)
        vla.to(DEVICE)
        return vla
    
# def get_vla(cfg):
#     """Loads and returns a VLA model from checkpoint."""
#     # Prepare for model loading.
#     print(f"[*] Initializing Generation Playground with `{cfg.model_family}`")
#     hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
#     set_seed(cfg.seed)
#     # Load VLA checkpoint.
#     print(f"Loading VLM from checkpoint: {cfg.pretrained_checkpoint}")
#     vla = load_vla(cfg.pretrained_checkpoint, hf_token=hf_token, load_for_training=False)
#     for param in vla.parameters():
#         assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"
#     # Cast to half precision.
#     vla.vision_backbone.to(dtype=vla.vision_backbone.half_precision_dtype)
#     vla.llm_backbone.to(dtype=vla.llm_backbone.half_precision_dtype)
#     vla.to(dtype=vla.llm_backbone.half_precision_dtype)
#     vla.to(DEVICE)
#     return vla


def get_model(cfg):
    """Load model for evaluation."""
    if cfg.model_family == "llava":
        model = get_vla(cfg)
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    print(f"Loaded model: {type(model)}")
    return model


def get_next_task_label(task_label):
    """Prompt the user to input the next task."""
    if task_label == "":
        user_input = ""
        while user_input == "":
            user_input = input("Enter the task name: ")
        task_label = user_input
    else:
        user_input = input("Enter the task name (or leave blank to repeat the previous task): ")
        if user_input == "":
            pass  # do nothing, let task_label be the same
        else:
            task_label = user_input
    print(f"Task: {task_label}")
    return task_label


def save_rollout_gif(rollout_images, idx):
    """Saves a GIF of an episode."""
    os.makedirs("./rollouts", exist_ok=True)
    gif_path = f"./rollouts/rollout-{DATE_TIME}-{idx}.gif"
    imageio.mimsave(gif_path, rollout_images, loop=0)
    print(f"Saved rollout GIF at path {gif_path}")
    # Save as mp4
    # mp4_path = f"./rollouts/rollout-{DATE_TIME}-{idx}.mp4"
    # imageio.mimwrite(mp4_path, rollout_images, fps=5)
    # print(f"Saved rollout MP4 at path {mp4_path}")



def get_preprocessed_image(obs, resize_size):
    """
    Extracts image from observations and preprocesses it.

    Preprocess the image the exact same way that the Berkeley Bridge folks did it
    to minimize distribution shift.
    NOTE (Moo Jin): Yes, we resize down to 256x256 first even though the image may end up being
                    resized up to a different resolution by some models. This is just so that we're in-distribution
                    w.r.t. the original preprocessing at train time.
    """
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    if len(obs["full_image"].shape) == 4:  # history included
        num_images_in_history = obs["full_image"].shape[0]
        assert resize_size[0] >= resize_size[1]  # in PIL format: (W, H) where W >= H
        W, H = resize_size
        new_images = np.zeros((num_images_in_history, H, W, obs["full_image"].shape[-1]), dtype=np.uint8)
        for i in range(num_images_in_history):
            new_images[i] = resize_image(obs["full_image"][i], resize_size)
        obs["full_image"] = new_images
    else:  # no history
        obs["full_image"] = resize_image(obs["full_image"], resize_size)
    return obs["full_image"]


def get_octo_policy_function(model):
    """Returns a JAX JIT-compiled Octo policy function."""
    import jax

    # create policy function
    @jax.jit
    def sample_actions(
        pretrained_model,
        observations,
        tasks,
        rng,
    ):
        # add batch dim to observations
        observations = jax.tree_map(lambda x: x[None], observations)
        actions = pretrained_model.sample_actions(
            observations,
            tasks,
            rng=rng,
        )
        # remove batch dim
        return actions[0]

    def supply_rng(f, rng):
        def wrapped(*args, **kwargs):
            nonlocal rng
            rng, key = jax.random.split(rng)
            return f(*args, rng=key, **kwargs)

        return wrapped

    policy_fn = supply_rng(
        partial(
            sample_actions,
            model,
        ),
        rng=jax.random.PRNGKey(0),
    )

    return policy_fn


# def get_vla_action(vla, obs, task_label, **kwargs):
#     """Generates an action with the VLA policy."""
#     image = Image.fromarray(obs["full_image"])
#     image = image.convert("RGB")
#     assert image.size[0] == image.size[1]
#     action = vla.predict_action(image, task_label, do_sample=False, **kwargs)
#     return action


def get_vla_action(
    vla,
    obs,
    task_label: str,
    **kwargs,
):
    """
    Generates an action with the VLA policy.
    - .pt 모델: 기존과 동일하게 PIL Image + task_label을 그대로 predict_action에 전달
    - HF 포맷 모델: processor를 통해 PIL 이미지와 instruction을 처리하도록 래핑 후 호출
    """
    # 1) PIL RGB 이미지로 변환
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")
    assert image.size[0] == image.size[1], "input image must be square"

    # 2) HF 포맷 모델인지 판단 (vla.processor 존재 여부)
    processor = getattr(vla, "processor", None)

    # === Case A: .pt 모델 ===
    if processor is None:
        # 기존 .pt 모델 방식 호출
        action = vla.predict_action(image, task_label, do_sample=False, **kwargs)
        return action

    # === Case B: HF 포맷 모델 ===
    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    info_dict = kwargs.pop("info_dict", None)

    # Process inputs.
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    out = vla.predict_action(**inputs, do_sample=False)
    if isinstance(out, tuple):
        action, generated_ids = out
    else:
        action = out
        generated_ids = None
    if(info_dict is not None) and (generated_ids is not None):
        tokenizer = getattr(vla, "tokenizer", None)
        if tokenizer is None and hasattr(vla, "processor"):
            tokenizer = getattr(vla.processor, "tokenizer", None)
        if tokenizer is not None:
            
        
    return action

def get_action(cfg, model, obs, task_label, policy_function=None, **kwargs):
    """Queries the model to get an action."""
    if cfg.model_family == "llava":
        action = get_vla_action(model, obs, task_label, **kwargs)
        assert action.shape == (ACTION_DIM,)
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return action


def refresh_obs(obs, env):
    """Fetches new observations from the environment and updates the current observations."""
    new_obs = env.get_observation()
    history_included = len(obs["full_image"].shape) == 4
    if history_included:
        obs["full_image"][-1] = new_obs["full_image"]
        obs["image_primary"][-1] = new_obs["image_primary"]
        obs["proprio"][-1] = new_obs["proprio"]
    else:
        obs["full_image"] = new_obs["full_image"]
        obs["image_primary"] = new_obs["image_primary"]
        obs["proprio"] = new_obs["proprio"]
    return obs




def split_reasoning(text, tags):
    new_parts = {None: text}

    for tag in tags:
        parts = new_parts
        new_parts = dict()

        for k, v in parts.items():
            if tag in v:
                s = v.split(tag)
                new_parts[k] = s[0]
                new_parts[tag] = s[1]
            else:
                new_parts[k] = v

    return new_parts


def get_metadata(reasoning):
    metadata = {"gripper": [[0, 0]], "bboxes": dict()}

    if f" {CotTag.GRIPPER_POSITION.value}" in reasoning:
        gripper_pos = reasoning[f" {CotTag.GRIPPER_POSITION.value}"]
        gripper_pos = gripper_pos.split("[")[-1]
        gripper_pos = gripper_pos.split("]")[0]
        gripper_pos = [int(x) for x in gripper_pos.split(",")]
        gripper_pos = [(gripper_pos[2 * i], gripper_pos[2 * i + 1]) for i in range(len(gripper_pos) // 2)]
        metadata["gripper"] = gripper_pos

    if f" {CotTag.VISIBLE_OBJECTS.value}" in reasoning:
        for sample in reasoning[f" {CotTag.VISIBLE_OBJECTS.value}"].split("]"):
            obj = sample.split("[")[0]
            if obj == "":
                continue
            coords = [int(n) for n in sample.split("[")[-1].split(",")]
            metadata["bboxes"][obj] = coords

    return metadata

# 원본 크기 기준 좌표를 현재 이미지 크기에 맞게 스케일 조정
def resize_pos(pos, img_shape, original_size=(224, 224)):
    h, w = img_shape[:2]
    return [
        int(pos[0] * w / original_size[0]),
        int(pos[1] * h / original_size[1]),
    ]

# 그리퍼 위치를 원 크기 기준으로 img에 그림
def draw_gripper(img, pos_list, original_size=(224, 224)):
    for i, pos in enumerate(reversed(pos_list)):
        x, y = resize_pos(pos, img.shape, original_size)
        scale = 255 - int(255 * i / len(pos_list))
        cv2.circle(img, (x, y), 6, (0, 0, 0), -1)
        cv2.circle(img, (x, y), 5, (scale, scale, 255), -1)

# "Interactive" 표시
def draw_interactive(img, is_interactive):
    h, _ = img.shape[:2]
    y = min(h - 30, 450)
    if is_interactive:
        cv2.putText(img, "Interactive", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, "Interactive", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

# 박스 색상 고정 방식
def name_to_random_color(name):
    return [(hash(name) // (256**i)) % 256 for i in range(3)]

# 바운딩 박스와 텍스트 이름 표시
def draw_bboxes(img, bboxes, original_size=(224, 224)):
    for name, bbox in bboxes.items():
        x1, y1 = resize_pos((bbox[0], bbox[1]), img.shape, original_size)
        x2, y2 = resize_pos((bbox[2], bbox[3]), img.shape, original_size)
        cv2.rectangle(img, (x1, y1), (x2, y2), name_to_random_color(name), 2)
        cv2.putText(
            img,
            name,
            (x1, y1 + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        
# def write_text(image, text, size, location, line_max_length):
#     next_x, next_y = location

#     for line in text:
#         x, y = next_x, next_y

#         for i in range(0, len(line), line_max_length):
#             line_chunk = line[i : i + line_max_length]
#             cv2.putText(image, line_chunk, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 1, cv2.LINE_AA)
#             y += 18

#         next_y = max(y, next_y + 30)

# # reasoning 텍스트 이미지를 시각화한 후 실제 이미지 높이에 맞춰 resize
# def make_reasoning_image(text, target_height=224):
#     """
#     - text: reasoning 텍스트
#     - target_height: left 이미지의 높이와 맞추기 위한 기준 (ex. 224)
#     """
#     base = np.zeros((224, 360, 3), dtype=np.uint8)

#     tags = [f" {tag}" for tag in get_cot_tags_list()]
#     reasoning = split_reasoning(text, tags)
#     text_lines = [tag + reasoning[tag] for tag in tags[:-1] if tag in reasoning]

#     write_text(base, text_lines, 0.2, (10, 30), 75)

#     # height 기준으로 resize (비율 유지)
#     orig_h, orig_w = base.shape[:2]
#     new_width = int((orig_w / orig_h) * target_height)
#     resized = cv2.resize(base, (new_width, target_height), interpolation=cv2.INTER_AREA)

#     #return resized, get_metadata(reasoning)
#     return resized


def write_text(image, text, font_scale, location, line_max_length):
    x0, y0 = location
    # 실제 글자 높이/폭으로 줄 간격 계산
    (sample_w, sample_h), _ = cv2.getTextSize("Hg", cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    line_step = sample_h + max(6, int(6 * font_scale))   # 최소 여백 6px 보장

    y = y0
    for line in text:
        # 고정 글자수 기반 wrap
        for i in range(0, len(line), line_max_length):
            chunk = line[i:i+line_max_length]
            cv2.putText(image, chunk, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 255, 255), 1, cv2.LINE_AA)
            y += line_step
        # 태그 사이 추가 여백
        y += int(line_step * 0.3)

def make_reasoning_image(
    text,
    target_height=224,
    font_scale=0.4,
    line_max_length=75,
    panel_width_px=None,               # ← 원하는 가로 픽셀 직접 지정(없으면 자동)
    exclude_tags=(" VISIBLE OBJECTS",) # ← 제외할 태그(문구 앞 공백 포함)
):
    # 태그 분해
    tags = [f" {tag}" for tag in get_cot_tags_list()]
    reasoning = split_reasoning(text, tags)

    # 태그 필터링(원하면 VISIBLE OBJECTS 등 제외)
    text_lines = []
    for tag in tags[:-1]:
        if tag in reasoning:
            if exclude_tags and any(tag.startswith(ex) for ex in exclude_tags):
                continue
            text_lines.append(tag + reasoning[tag])

    # 실제 줄바꿈을 반영해 "그려질 줄 리스트" 미리 계산
    wrapped = []
    for line in text_lines:
        for i in range(0, len(line), line_max_length):
            wrapped.append(line[i:i+line_max_length])

    # 글자 픽셀 폭/높이 측정
    (char_w_est, char_h_est), _ = cv2.getTextSize("M", cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    line_step = char_h_est + max(6, int(6 * font_scale))

    # 패널 폭 계산: 직접 지정이 있으면 우선, 없으면 "가장 긴 조각" 기준 + 여백
    max_line_len = max((len(s) for s in wrapped), default=line_max_length)
    if panel_width_px is None:
        panel_width_px = max(360, 20 + max_line_len * char_w_est + 20)  # 좌우 여백 20px

    # 패널 높이 계산: 줄 수 × 줄 간격 + 상하 여백
    total_lines = len(wrapped) + int(0.3 * len(text_lines))  # 태그 간 여백 반영
    panel_height_px = max(224, 30 + total_lines * line_step + 30)

    # 캔버스 생성
    base = np.zeros((panel_height_px, panel_width_px, 3), dtype=np.uint8)

    # 텍스트 그리기
    write_text(base, text_lines, font_scale, (10, 30), line_max_length)

    # 최종 resize(높이만 target_height로 맞춤, 가로는 비율 유지)
    H, W = base.shape[:2]
    resized = cv2.resize(base, (panel_width_px, target_height), interpolation=cv2.INTER_AREA)
    return resized


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action


def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action
