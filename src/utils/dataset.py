import os
import random
import tensorflow as tf

IMG_SIZE = (224,224)
LABELS = {"real": 0, "fake": 1}

def split_by_video(data_dir, seed=42, val_frac=0.15, test_frac=0.15):
    rng=random.Random(seed)
    splits = {"train": ([], []), "val": ([], []), "test": ([], [])}

    for label_name, label in LABELS.items():
        class_dir = os.path.join(data_dir, label_name)
        videos = sorted(os.listdir(class_dir))
        rng.shuffle(videos)

        n=len(videos)
        n_val = max(1, round(n*val_frac))
        n_test = max(1, round(n*test_frac))
        n_train=n - n_val -n_test

        groups = {
            "train": videos[:n_train],
            "val": videos[n_train:n_train + n_val],
            "test": videos[n_train + n_val:],
        }

        for split_name, split_videos in groups.items():
            paths, labels = splits[split_name]
            for video in split_videos:
                video_dir = os.path.join(class_dir, video)
                for frame_file in sorted(os.listdir(video_dir)):
                    paths.append(os.path.join(video_dir, frame_file))
                    labels.append(float(label))
    return splits

def load_image(path, label):
    img=tf.io.read_file(path)
    img=tf.image.decode_png(img, channels=3)
    img=tf.image.resize(img, IMG_SIZE)
    return img, label

def build_dataset(paths, labels, batch_size, shuffle, seed):
    ds=tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def list_split_files(data_dir, seed=42):
    if not os.path.isdir(os.path.join(data_dir, "train")):
        return split_by_video(data_dir, seed=seed)

    splits = {}
    for split in ["train", "val", "test"]:
        paths, labels = [], []
        for label_name, label in LABELS.items():
            class_dir = os.path.join(data_dir, split, label_name)
            for video in sorted(os.listdir(class_dir)):
                video_dir = os.path.join(class_dir, video)
                for frame_file in sorted(os.listdir(video_dir)):
                    paths.append(os.path.join(video_dir, frame_file))
                    labels.append(float(label))
        splits[split] = (paths, labels)
    return splits


def get_datasets(data_dir, batch_size=16, seed=42):
    splits = list_split_files(data_dir, seed=seed)
    datasets=[]
    for name in ["train", "val", "test"]:
        paths, labels = splits[name]
        print(f"{name}:{len(paths)} imagens")
        datasets.append(build_dataset(paths, labels,batch_size, shuffle=(name=="train"), seed=seed))
    return datasets

