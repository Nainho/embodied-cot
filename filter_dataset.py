# ruff: noqa: B006
import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

tf.config.set_visible_devices([], device_type="GPU")

# ===== 경로/설정 =====
ORIG_DATASET_DIR = "/home/work/AGI_NIH/data/embodied_features_and_demos_libero/libero_90/1.0.0"
REASONING_PATH   = "/home/work/AGI_NIH/data/embodied_features_and_demos_libero/libero_reasonings.json"
SAVE_DIR_TFDS    = "/home/work/AGI_NIH/data/LIBERO/libero/datasets/"

# 새 데이터셋 버전 (필요 시 조정)
NEW_VERSION = "1.0.0"

# 3개 태스크(파일 basename) — 대소문자 무시 비교
ALLOWED_FILE_BASENAMES_LC = {
    "study_scene1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy_demo.hdf5",
    "study_scene2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy_demo.hdf5",
    "study_scene3_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy_demo.hdf5",
}


def _features_spec():
    # 원본 스키마 그대로
    return tfds.features.FeaturesDict({
        'episode_metadata': tfds.features.FeaturesDict({
            'demo_id': tf.int32,
            'file_path': tfds.features.Text(),
            'seg_labels': tfds.features.Text(),
        }),
        'steps': tfds.features.Dataset({
            'action':   tfds.features.Tensor(shape=(7,), dtype=tf.float32),
            'discount': tfds.features.Scalar(dtype=tf.float32),
            'is_first': tf.bool,
            'is_last':  tf.bool,
            'is_terminal': tf.bool,
            'language_instruction': tfds.features.Text(),
            'language_motions': tfds.features.Text(),
            'language_motions_future': tfds.features.Text(),
            'observation': tfds.features.FeaturesDict({
                'image':       tfds.features.Image(shape=(224, 224, 3), dtype=tf.uint8),
                'joint_state': tfds.features.Tensor(shape=(7,), dtype=tf.float32),
                'seg':         tfds.features.Image(shape=(224, 224, 1), dtype=tf.uint8),
                'state':       tfds.features.Tensor(shape=(8,), dtype=tf.float32),
                'wrist_image': tfds.features.Image(shape=(224, 224, 3), dtype=tf.uint8),
            }),
            'reward': tfds.features.Scalar(dtype=tf.float32),
        }),
    })


def _b(t):
    if isinstance(t, (bytes, np.bytes_)):
        return t.decode('utf-8')
    if isinstance(t, str):
        return t
    if hasattr(t, "numpy"):
        v = t.numpy()
        if isinstance(v, (bytes, np.bytes_)):
            return v.decode('utf-8')
        return str(v)
    return str(t)


class LiberoLm90_FullyReasoned_BookLeftCaddy(tfds.core.GeneratorBasedBuilder):
    """3개 태스크에 한정 + 모든 step reasoning이 있는 episode만 포함"""
    VERSION = tfds.core.Version(NEW_VERSION)

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            features=_features_spec(),
            description=(
                "Subset of libero_lm_90 containing only the 3 'pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy' tasks "
                "across scenes 1-3, where every step has a reasoning entry. Schema matches the original."
            ),
        )

    def _split_generators(self, dl_manager):
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "orig_dataset_dir": ORIG_DATASET_DIR,
                    "reasoning_json_path": REASONING_PATH,
                },
            ),
        ]

    def _generate_examples(self, orig_dataset_dir: str, reasoning_json_path: str):
        print(f"📦 Loading original TFDS: {orig_dataset_dir}")
        orig_builder = tfds.builder_from_directory(orig_dataset_dir)
        ds = orig_builder.as_dataset(split="train")

        print(f"📖 Loading reasoning json: {reasoning_json_path}")
        with tf.io.gfile.GFile(reasoning_json_path, "r") as f:
            reasoning_data = json.load(f)

        kept, skip_reasoning, skip_task, total = 0, 0, 0, 0

        for ep in ds:
            total += 1
            meta = ep["episode_metadata"]
            file_path = _b(meta["file_path"])
            base_lc = os.path.basename(file_path).lower()
            demo_id = str(int(meta["demo_id"].numpy()))
            steps_n = len(ep["steps"])

            # (1) 태스크 필터: 지정된 3개 파일 basename만
            if base_lc not in ALLOWED_FILE_BASENAMES_LC:
                skip_task += 1
                continue

            # (2) reasoning 전 step 커버리지 확인
            if not (
                (file_path in reasoning_data) and
                (demo_id in reasoning_data[file_path]) and
                all(str(i) in reasoning_data[file_path][demo_id] for i in range(steps_n))
            ):
                skip_reasoning += 1
                continue

            # (3) steps 복사 (원본 스키마 그대로)
            steps_list = []
            for st in ep["steps"]:
                obs = st["observation"]

                image = obs["image"].numpy()
                seg   = obs["seg"].numpy()
                joint = obs["joint_state"].numpy().astype(np.float32)
                state = obs["state"].numpy().astype(np.float32)

                # wrist_image가 스키마상 필수이므로, 누락 시 zero로 보정 (실제 데이터에 존재한다면 그대로 사용)
                if "wrist_image" in obs:
                    wrist = obs["wrist_image"].numpy()
                else:
                    wrist = np.zeros_like(image, dtype=np.uint8)

                steps_list.append({
                    "action":   st["action"].numpy().astype(np.float32),
                    "discount": float(st["discount"].numpy()),
                    "is_first": bool(st["is_first"].numpy()),
                    "is_last":  bool(st["is_last"].numpy()),
                    "is_terminal": bool(st["is_terminal"].numpy()),
                    "language_instruction": _b(st["language_instruction"]),
                    "language_motions": _b(st["language_motions"]),
                    "language_motions_future": _b(st["language_motions_future"]),
                    "observation": {
                        "image":       image,
                        "joint_state": joint,
                        "seg":         seg,
                        "state":       state,
                        "wrist_image": wrist,
                    },
                    "reward": float(st["reward"].numpy()),
                })

            example = {
                "episode_metadata": {
                    "demo_id":   int(meta["demo_id"].numpy()),
                    "file_path": file_path,
                    "seg_labels": _b(meta["seg_labels"]),
                },
                "steps": steps_list,
            }

            key = f"{os.path.basename(file_path)}__demo{demo_id}"
            kept += 1
            yield key, example

        print("📊 Build summary")
        print(f"  - total episodes scanned : {total}")
        print(f"  - kept                   : {kept}")
        print(f"  - skipped (task filter)  : {skip_task}")
        print(f"  - skipped (reasoning)    : {skip_reasoning}")


if __name__ == "__main__":
    # 데이터셋 이름은 클래스명을 snake_case로 변환해서 생성됩니다:
    # LiberoLm90_FullyReasoned_BookLeftCaddy -> "libero_lm90_fully_reasoned_book_left_caddy"
    builder = LiberoLm90_FullyReasoned_BookLeftCaddy(data_dir=SAVE_DIR_TFDS)
    print("🛠 Writing filtered TFDS dataset...")
    builder.download_and_prepare()
    print("🎉 Done.")
    print("📚 New dataset info:")
    print(builder.info)
    print(f"📁 Saved under: {os.path.join(SAVE_DIR_TFDS, builder.info.full_name)}")




"""
📖 Loading reasoning json: /home/work/AGI_NIH/data/embodied_features_and_demos_libero/libero_reasonings.json
                                                                    📊 Build summary
  - total episodes scanned : 3917 examples [11:19,  5.30s/ examples]
  - kept                   : 129
  - skipped (task filter)  : 3780
  - skipped (reasoning)    : 8
Dataset libero_lm90__fully_reasoned__book_left_caddy downloaded and prepared to /home/work/AGI_NIH/data/LIBERO/libero/datasets/libero_lm90__fully_reasoned__book_left_caddy/1.0.0. Subsequent calls will reuse this data.                              
🎉 Done.                                                                                                                                                                                                                                               
📚 New dataset info:
tfds.core.DatasetInfo(
    name='libero_lm90__fully_reasoned__book_left_caddy',
    full_name='libero_lm90__fully_reasoned__book_left_caddy/1.0.0',
    description="""
    Subset of libero_lm_90 containing only the 3 'pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy' tasks across scenes 1-3, where every step has a reasoning entry. Schema matches the original.
    """,
    homepage='https://www.tensorflow.org/datasets/catalog/libero_lm90__fully_reasoned__book_left_caddy',
    data_dir='/home/work/AGI_NIH/data/LIBERO/libero/datasets/libero_lm90__fully_reasoned__book_left_caddy/1.0.0',
    file_format=tfrecord,
    download_size=Unknown size,
    dataset_size=1.65 GiB,
    features=FeaturesDict({
        'episode_metadata': FeaturesDict({
            'demo_id': int32,
            'file_path': Text(shape=(), dtype=string),
            'seg_labels': Text(shape=(), dtype=string),
        }),
        'steps': Dataset({
            'action': Tensor(shape=(7,), dtype=float32),
            'discount': Scalar(shape=(), dtype=float32),
            'is_first': bool,
            'is_last': bool,
            'is_terminal': bool,
            'language_instruction': Text(shape=(), dtype=string),
            'language_motions': Text(shape=(), dtype=string),
            'language_motions_future': Text(shape=(), dtype=string),
            'observation': FeaturesDict({
                'image': Image(shape=(224, 224, 3), dtype=uint8),
                'joint_state': Tensor(shape=(7,), dtype=float32),
                'seg': Image(shape=(224, 224, 1), dtype=uint8),
                'state': Tensor(shape=(8,), dtype=float32),
                'wrist_image': Image(shape=(224, 224, 3), dtype=uint8),
            }),
            'reward': Scalar(shape=(), dtype=float32),
        }),
    }),
    supervised_keys=None,
    disable_shuffling=False,
    splits={
        'train': <SplitInfo num_examples=129, num_shards=16>,
    },
    citation="""""",
)
📁 Saved under: /home/work/AGI_NIH/data/LIBERO/libero/datasets/libero_lm90__fully_reasoned__book_left_caddy/1.0.0
"""