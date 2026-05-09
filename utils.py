from __future__ import annotations

from pathlib import Path
from datasets import load_dataset, load_from_disk
import boto3
import torch
import yaml
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig

#Load Experiment Config
with open("experiment_config.yaml", "r") as file:
    experiment_config = yaml.safe_load(file)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def s3_sync_dir_to_prefix(local_dir: str, bucket: str, prefix: str) -> None:
    """
    Upload all files under local_dir to s3://bucket/prefix/ preserving relative paths.
    """
    s3 = boto3.client("s3")
    local_dir_path = Path(local_dir).resolve()

    for path in local_dir_path.rglob("*"):
        if path.is_dir():
            continue
        rel_key = str(path.relative_to(local_dir_path)).replace("\\", "/")
        key = f"{prefix.rstrip('/')}/{rel_key}"
        s3.upload_file(str(path), bucket, key)


def s3_download_prefix_to_dir(prefix: str, local_dir: str,bucket : str) -> None:
    """
    Downloads all files under local_dir to s3://bucket/prefix/ preserving relative paths.
    :param prefix:
    :param local_dir:
    :param bucket:
    :return:
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    prefix = prefix.lstrip("/").rstrip("/") + "/"
    local_dir_path = Path(local_dir)
    local_dir_path.mkdir(parents=True, exist_ok=True)

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue

            rel = key[len(prefix):]
            if not rel:
                continue

            out_path = local_dir_path / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)

            s3.download_file(bucket, key, str(out_path))

def preprocess(example, tokenizer,max_input_length):
    """
    This converts text into tensors the model can consume
    :param example:
    :param tokenizer:
    :param max_input_length:
    :return:
    """
    inputs = tokenizer(
        example["article"], max_length=max_input_length, truncation=True, return_tensors="pt"
    ).to(device)
    return inputs, example["highlights"]


def load_cnn_dm_split(split: str, local_cache="../data/cnn_dm_splits_v1"):
    """
    Loads the CNN DM split, e.g train, dev, test
    :param split:
    :param local_cache:
    :return:
    """
    bucket = experiment_config["dataset"]["s3_bucket"]
    prefix = f"cnn_dm_splits_v1/{split}"

    local_dir = f"{local_cache}/{split}"

    if not Path(local_dir).exists():
        s3_download_prefix_to_dir(prefix, local_dir,bucket)

    return load_from_disk(local_dir)


def get_model_size_mb(model):
    total_bytes = 0
    for name, param in model.named_parameters():
        total_bytes += param.numel() * param.element_size()

    for name, buffer in model.named_buffers():
        total_bytes += buffer.numel() * buffer.element_size()

    return total_bytes / (1024 ** 2)


def load_model(model_source: str,device):
    model_path = Path(model_source)

    if model_path.exists():
        print("Loading local model...")
        config = AutoConfig.from_pretrained(model_source)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path), AutoTokenizer.from_pretrained(model_path)
        model.to(device)
        return model, AutoTokenizer.from_pretrained(model_path), config

    else:
        print("Downloading model from Hugging Face...")
        config = AutoConfig.from_pretrained(model_source)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_source,
        )
        model.to(device)
        return model, AutoTokenizer.from_pretrained(model_source), config


def load_model_quantized(model_id: str,device,quantization_config):
    print("Downloading quantized model from Hugging Face...")

    if quantization_config["weight_bits"]=="4bit":
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    elif quantization_config["weight_bits"]=="8bit":
        config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
            bnb_8bit_quant_type="nf8",
        )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        quantization_config=config,
        device_map=device,  # single GPU;
        low_cpu_mem_usage=False,
    )
    model.generation_config.forced_bos_token_id = 0
    return model, AutoTokenizer.from_pretrained(model_id), config

