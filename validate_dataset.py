# ruff: noqa: B006
import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

tf.config.set_visible_devices([], device_type="GPU")

# ===== 경로 설정 =====
DATASET_DIR   = "/home/work/AGI_NIH/data/embodied_features_and_demos_libero/libero_90/1.0.0"
REASONING_PATH = "/home/work/AGI_NIH/data/embodied_features_and_demos_libero/libero_reasonings.json"

# ===== 태스크(파일 basename, 소문자) 필터 집합 =====
ALLOWED_FILE_BASENAMES_LC = {
    "study_scene1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy_demo.hdf5",
    "study_scene2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy_demo.hdf5",
    "study_scene3_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy_demo.hdf5",
}

# ===== 기대 스키마 (shape/dtype) =====
EXPECTED = {
    "action": (tf.float32, (7,)),
    "discount": (tf.float32, ()),
    "is_first": (tf.bool, ()),
    "is_last": (tf.bool, ()),
    "is_terminal": (tf.bool, ()),
    "language_instruction": (tf.string, ()),
    "language_motions": (tf.string, ()),
    "language_motions_future": (tf.string, ()),
    "observation.image": (tf.uint8, (224, 224, 3)),
    "observation.wrist_image": (tf.uint8, (224, 224, 3)),
    "observation.seg": (tf.uint8, (224, 224, 1)),
    "observation.joint_state": (tf.float32, (7,)),
    "observation.state": (tf.float32, (8,)),
    "reward": (tf.float32, ()),
}
EXPECTED_META = {
    "episode_metadata.demo_id": tf.int32,
    "episode_metadata.file_path": tf.string,
    "episode_metadata.seg_labels": tf.string,
}


def _shape_tuple(ts):
    return tuple(ts.shape.as_list()) if hasattr(ts, "shape") else None


def validate_dataset_schema(dataset_dir: str) -> None:
    print(f"📦 Loading TFDS from: {dataset_dir}")
    builder = tfds.builder_from_directory(dataset_dir)
    print("🔎 Declared features (from dataset_info):")
    print(builder.info.features)

    ds = builder.as_dataset(split="train")

    errors = []
    ep_cnt = 0
    step_cnt = 0

    for ep in ds:
        ep_cnt += 1
        # --- metadata dtype 체크
        for k, exp_dtype in EXPECTED_META.items():
            *head, leaf = k.split(".")
            cur = ep
            for key in head:
                cur = cur[key]
            cur = cur[leaf]
            if cur.dtype != exp_dtype:
                errors.append(f"[META] {k} dtype={cur.dtype} expected={exp_dtype}")

        # --- steps 전수 검사
        for st in ep["steps"]:
            step_cnt += 1
            # 평판/스칼라/이미지/텍스트 모두 shape·dtype 체크
            for k, (exp_dtype, exp_shape) in EXPECTED.items():
                cur = st
                for part in k.split("."):
                    cur = cur[part]
                # dtype
                if cur.dtype != exp_dtype:
                    errors.append(f"[STEP] {k} dtype={cur.dtype} expected={exp_dtype}")
                # shape (스칼라는 () 로 고정)
                got_shape = _shape_tuple(cur)
                if got_shape != exp_shape:
                    errors.append(f"[STEP] {k} shape={got_shape} expected={exp_shape}")

    print(f"✅ Episodes scanned: {ep_cnt}, Steps scanned: {step_cnt}")
    if errors:
        print(f"❌ MISMATCHES: {len(errors)} (showing first 30)")
        for e in errors[:30]:
            print("  -", e)
    else:
        print("🎉 SCHEMA OK — all dtypes/shapes match the spec.")


def validate_reasoning_coverage(dataset_dir: str, reasoning_json_path: str) -> None:
    print(f"📖 Loading reasoning json: {reasoning_json_path}")
    with tf.io.gfile.GFile(reasoning_json_path, "r") as f:
        reasoning_data = json.load(f)

    builder = tfds.builder_from_directory(dataset_dir)
    ds = builder.as_dataset(split="train")

    total_eps = 0
    full_reason_eps = 0
    task_eps = 0
    task_full_reason_eps = 0

    for ep in ds:
        total_eps += 1
        meta = ep["episode_metadata"]
        file_path = meta["file_path"].numpy().decode("utf-8")
        demo_id = str(int(meta["demo_id"].numpy()))
        base_lc = os.path.basename(file_path).lower()

        steps_n = len(ep["steps"])
        has_full_reason = (
            (file_path in reasoning_data) and
            (demo_id in reasoning_data[file_path]) and
            all(str(i) in reasoning_data[file_path][demo_id] for i in range(steps_n))
        )

        in_task = base_lc in ALLOWED_FILE_BASENAMES_LC
        if in_task:
            task_eps += 1
            if has_full_reason:
                task_full_reason_eps += 1

        if has_full_reason:
            full_reason_eps += 1

    print("📊 Reasoning coverage summary")
    print(f"  - Episodes total:                 {total_eps}")
    print(f"  - Fully reasoned episodes:        {full_reason_eps}")
    print(f"  - Episodes in 3-book tasks:       {task_eps}")
    print(f"  - Fully reasoned within 3 tasks:  {task_full_reason_eps}")


if __name__ == "__main__":
    validate_dataset_schema(DATASET_DIR)
    print("-" * 80)
    validate_reasoning_coverage(DATASET_DIR, REASONING_PATH)


"""
✅ Episodes scanned: 3917, Steps scanned: 567494
🎉 SCHEMA OK — all dtypes/shapes match the spec.
--------------------------------------------------------------------------------
📖 Loading reasoning json: /home/work/AGI_NIH/data/embodied_features_and_demos_libero/libero_reasonings.json
📊 Reasoning coverage summary
  - Episodes total:                 3917
  - Fully reasoned episodes:        3435
  - Episodes in 3-book tasks:       137
  - Fully reasoned within 3 tasks:  129
"""