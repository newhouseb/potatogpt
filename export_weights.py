# This script can be used to extract weights from the bananas-complicated tensorflow checkpoint for GPT2 and just
# save them to disk as packed float32 arrays. Note, thought, that the weights are already part of the repo, so this
# script is largely included for reproducability purposes. Since running tensorflow on a local M1/M2 Mac is a pain,
# if you're using an M1/M2 mac, you're probably better off running this in something like colab or a docker container.
 
# The first half of this is adapted from https://github.com/jaymody/picoGPT/blob/main/utils.py

import json
import os
import re
import struct
import shutil

import numpy as np
import requests
import tensorflow as tf
from tqdm import tqdm

def download_gpt2_files(model_size, model_dir):
    assert model_size in ["124M", "355M", "774M", "1558M"]
    for filename in [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]:
        url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
        r = requests.get(f"{url}/{model_size}/{filename}", stream=True)
        r.raise_for_status()

        with open(os.path.join(model_dir, filename), "wb") as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(
                ncols=100,
                desc="Fetching " + filename,
                total=file_size,
                unit_scale=True,
                unit="b",
            ) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)


def load_gpt2_params_from_tf_ckpt(tf_ckpt_path, hparams):
    def set_in_nested_dict(d, keys, val):
        if not keys:
            return val
        if keys[0] not in d:
            d[keys[0]] = {}
        d[keys[0]] = set_in_nested_dict(d[keys[0]], keys[1:], val)
        return d

    params = {"blocks": [{} for _ in range(hparams["n_layer"])]}
    for name, _ in tf.train.list_variables(tf_ckpt_path):
        array = np.squeeze(tf.train.load_variable(tf_ckpt_path, name))
        name = name[len("model/") :]
        if name.startswith("h"):
            m = re.match(r"h([0-9]+)/(.*)", name)
            n = int(m[1])
            sub_name = m[2]
            set_in_nested_dict(params["blocks"][n], sub_name.split("/"), array)
        else:
            set_in_nested_dict(params, name.split("/"), array)

    return params


def load_encoder_hparams_and_params(model_size, models_dir):
    assert model_size in ["124M", "355M", "774M", "1558M"]

    model_dir = os.path.join(models_dir, model_size)
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    if not tf_ckpt_path:  # download files if necessary
        os.makedirs(model_dir, exist_ok=True)
        download_gpt2_files(model_size, model_dir)
        tf_ckpt_path = tf.train.latest_checkpoint(model_dir)

    hparams = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, hparams)

    return hparams, params


# Download the small model and traverse various tensors and write one tensor per file

os.makedirs('sm', exist_ok=True)
hparams, params = load_encoder_hparams_and_params('124M', 'sm')

def convert_nested_list_to_float32_array(nested_list) -> np.ndarray:
    float32_array = np.array(nested_list).flatten() 
    bytearray_float32 = bytearray(4 * len(float32_array))
    for i, num in enumerate(float32_array):
        bytearray_float32[4*i : 4*i + 4] = struct.pack('f', num)

    return bytearray_float32 

def traverse(d, prefix=''):
    for k, v in d.items():
        if k == 'blocks':
            for i, block in enumerate(v):
                traverse(block, prefix + 'blocks_' + str(i) + '_')
        elif isinstance(v, dict):
            traverse(v, prefix + k + '_')
        else:
            v = convert_nested_list_to_float32_array(v)
            print("Writing", prefix + k, len(v))
            with open('weights/' + prefix + k, 'wb') as f:
                f.write(v)

shutil.copyfile('sm/124M/encoder.json', 'weights/encoder.json')
os.makedirs('weights', exist_ok=True)
traverse(params)

# Github limits files to 100MB, so we split the wte file into two chunks to avoid
# having to deal with Git LFS

def split_file(file_path):
    file_size = os.path.getsize(file_path)
    chunk_size = file_size // 2
    with open(file_path, 'rb') as input_file:
        chunk1_data = input_file.read(chunk_size)
        with open(file_path + '.0', 'wb') as chunk1_file:
            chunk1_file.write(chunk1_data)
        chunk2_data = input_file.read()
        with open(file_path + '.1', 'wb') as chunk2_file:
            chunk2_file.write(chunk2_data)

file_path = 'weights/wte'
split_file(file_path)
