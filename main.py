"""
Parameters:
- dataset_name: the dataset to find duplicates for (e.g., "cifar100", "caltech101", etc.)
- threshold(inclusive): the maximum distance for search results, in hamming distance
- k: the number of results to show
NOTE: need to provide classes as a list if labels are numerical ELSE LEFT THE CLASSES LIST EMPTY
"""

import os
import glob
import json
from pathlib import Path
from PIL import Image
import imagehash
import faiss
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, Tuple, List
import torch
import torch.nn.functional as F
from lightning_cloud.utils.data_connection import add_s3_connection
from utils.laion_streaming_dataset import LAOINStreamingDataset
from utils.HF_dataset_eval import HFDataset_eval
from utils.HF_dataset import HFDataset
from utils.optimize_hf_to_lightning import optimize_hf_to_lightning
import clip
import textwrap
from openai import OpenAI

# ============= Parameter Configuration =============
# Dataset Configuration
dataset_name = "cifar100"  # Options: "cifar100", "caltech101", "food101", "cars", "country211", 
                          # "sun397", "fer2013", "aircraft", "imagenetv2", "imagenet-o", "pets", 
                          # "imagenet-a", "imagenet-r", "cub"

# Search Parameters
threshold = 4  # Maximum Hamming distance for considering images as duplicates (inclusive)
k = 5  # Number of duplicate matches to show per image

# Processing Parameters
batch_size = 64  # Batch size for processing images
num_workers = 4  # Number of worker processes for data loading
sim_threshold = 0.88  # CLIP similarity threshold for filtering duplicates
# ================================================

# Load classes if needed
try:
    classes = json.load(open(f"data/classes/classes_{dataset_name}.json", "r"))
    print(f"Number of classes: {len(classes)}")
except FileNotFoundError:
    classes = None
    print("No classes file found, using raw labels")

# Load the target dataset from Huggingface
if dataset_name == "caltech101":
    hf_dataset = load_dataset("clip-benchmark/wds_vtab-caltech101")
elif dataset_name == "food101":
    hf_dataset = load_dataset("clip-benchmark/wds_food101")
elif dataset_name == "cars":
    hf_dataset = load_dataset("clip-benchmark/wds_cars")
elif dataset_name == "country211":
    hf_dataset = load_dataset("clip-benchmark/wds_country211")
elif dataset_name == "sun397":
    hf_dataset = load_dataset("clip-benchmark/wds_sun397")
elif dataset_name == "fer2013":
    hf_dataset = load_dataset("clip-benchmark/wds_fer2013")
elif dataset_name == "aircraft":
    hf_dataset = load_dataset("clip-benchmark/wds_fgvc_aircraft")
elif dataset_name == "imagenetv2":
    hf_dataset = load_dataset("clip-benchmark/wds_imagenetv2")
elif dataset_name == "imagenet-o":
    hf_dataset = load_dataset("clip-benchmark/wds_imagenet-o")
elif dataset_name == "pets":
    hf_dataset = load_dataset("clip-benchmark/wds_vtab-pets")
elif dataset_name == "imagenet-a":
    hf_dataset = load_dataset("clip-benchmark/wds_imagenet-a")
elif dataset_name == "imagenet-r":
    hf_dataset = load_dataset("clip-benchmark/wds_imagenet-r")
elif dataset_name == "cub":
    hf_dataset = load_dataset("lxs784/cub-200-2011-clip-benchmark")
elif dataset_name == "cifar100":
    hf_dataset = load_dataset("clip-benchmark/wds_vtab-cifar100")
else:
    raise ValueError(f"Unsupported dataset: {dataset_name}")

# Determine image key using the first available split
first_split = next(iter(hf_dataset.keys()))
if "webp" in hf_dataset[first_split][0] and hf_dataset[first_split][0]["webp"] is not None:
    image_key = "webp"
elif hf_dataset[first_split][0]["jpg"] is not None:
    image_key = "jpg"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

with open("api.json", "r") as f:
    api_key = json.load(f)["api_key"]
client = OpenAI(api_key=api_key)

def hex_to_vector(hex_str, vector_dim=16):
    """
    Convert a 16-character hex string to a 64-bit binary vector. 
    Each hex digit is converted to a 4-bit binary number.
    Ensure the hex string is exactly 16 characters long and binary vector is exactly 64 bits long.
    """
    if hex_str is None:
        return [0] * vector_dim * 4

    if len(hex_str) != vector_dim:
        raise ValueError(f"Hex string length ({len(hex_str)}) does not match expected dimension ({vector_dim}).")
    
    vector = []
    for digit in hex_str:
        if digit not in "0123456789abcdef":
            raise ValueError("Invalid hex string")

        binary_str = bin(int(digit, 16))[2:].zfill(4)
        vector.extend([int(bit) for bit in binary_str])

    if len(vector) != vector_dim * 4:
        raise ValueError("Hex string did not convert to the expected number of bits")
    return vector

def find_duplicates(dataset_name: str, dataloader, threshold: int, binary_index_phash, k: int = 5) -> dict:
    """
    Find duplicate images between target dataset and LAION-400M.
    Parameters:
        dataset_name (str): The name of the dataset to process.
        dataloader (iterable): An iterable (e.g., a DataLoader) that yields batches of data.
            Each batch is expected to be a tuple containing:
                - an ignored element (e.g., image data),
                - texts (list or similar),
                - ahashes (list or similar),
                - phashes (list of hexadecimal pHash strings),
                - uids (list of unique identifiers for each image).
        threshold (int): The maximum Hamming distance to consider two images as duplicates.
        binary_index_phash: An object with a method `range_search` that takes a packed query array and threshold,
            and returns search results (lims, D_range, I_range) for duplicate detection.
        hex_to_vector (callable): A function that converts a hexadecimal pHash string into a vector of integers.
        k (int, optional): Maximum number of duplicate matches to retain per image. Defaults to 5.

    Returns:
        dict: A dictionary mapping image unique IDs (uids) to a list of duplicate indices found.
              Also saves intermediate and combined results in JSON files under the designated directory.
    """
    results = {}
    part = 0
    json_dir = f"/teamspace/studios/this_studio/data/intermediate/{dataset_name}/match_indices_{threshold}"
    os.makedirs(json_dir, exist_ok=True)

    for i, (_, texts, ahashes, phashes, uids) in enumerate(tqdm(dataloader, desc=f"Finding duplicates in {dataset_name}")):
        query_vectors = np.array([hex_to_vector(x, 16) for x in phashes], dtype='uint8')
        queries_packed = np.packbits(query_vectors, axis=1).reshape(len(phashes), 8)

        lims, D_range, I_range = binary_index_phash.range_search(queries_packed, threshold)

        for q in range(queries_packed.shape[0]):
            start = lims[q]
            end = lims[q + 1]
            if start == end:
                continue
            match_indices = I_range[start:end].tolist()
            if len(match_indices) > 0:
                results[uids[q]] = match_indices
            if len(results) == 100:
                with open(os.path.join(json_dir, f"results_{part}.json"), "w") as f:
                    json.dump(results, f)
                    tqdm.write(f"part {part} saved!")
                results = {} # reset
                part += 1
                
    if len(results) > 0:
        with open(os.path.join(json_dir, f"results_{part}.json"), "w") as f:
            json.dump(results, f)

    # put all results in one json file
    json_files = glob.glob(os.path.join(json_dir, "*.json"))

    results = {}
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)
        results.update(data)

    with open(os.path.join(json_dir, "combined_results.json"), "w") as f:
        json.dump(results, f)
    print(f"Combined results saved to {os.path.join(json_dir, 'combined_results.json')}, total duplicate images: {len(results)}")
    return results

def resize_image(image, target_size=(256, 256)):
    return image.resize(target_size, Image.Resampling.LANCZOS)

def visualize_duplicates(dataset_name: str, dataset: HFDataset, results: dict, laion, k: int = k):
    """
    Visualize duplicate pairs of images between target dataset and LAION-400M.
    
    Parameters:
        target_dataset: The dataset to find duplicates for
        laion_dataset: The LAION-400M dataset
        duplicates: List of (target_uid, laion_uid, distance) tuples
        save_dir: Directory to save visualization images
    """
    output_dir = f"data/intermediate/{dataset_name}/plots"
    os.makedirs(output_dir, exist_ok=True)

    cols = k + 2
    for uid, match_indices in tqdm(results.items(), desc=f"plotting duplicate images for {dataset_name}"):
        fig, axes = plt.subplots(1, cols, figsize=(cols * 3, 3))
        axes[0].text(0.5, 0.5, uid, fontsize=24, ha='center', va='center')
        axes[0].axis("off")

        original_image, original_text, ahash, phash= dataset.get_by_id(uid)
        original_image_resized = resize_image(original_image)
        axes[1].imshow(original_image_resized)
        wrapped_caption = "\n".join(textwrap.wrap(original_text, width=24))
        axes[1].set_title(wrapped_caption)
        axes[1].axis('off')

        for j in range (k):
            ax = axes[j + 2]
            if j >= len(match_indices):
                ax.imshow(np.ones((1, 1, 3)))
            else:
                idx = match_indices[j]
                match_image, match_text, _ = laion[idx]
                laion_phash = imagehash.phash(match_image)
                p_dist = abs(phash - laion_phash)
                ax.imshow(match_image)
                caption_match = "p_dist: " + str(p_dist) + " " + match_text
                wrapped_lines = textwrap.wrap(caption_match, width=24)
                wrapped_caption_match = "\n".join(wrapped_lines[:2])
                ax.set_title(wrapped_caption_match, fontsize=8)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{uid}.png"))
        plt.close(fig)
    print(f"Visualizations saved to {output_dir}")

def filter_and_visualize_duplicates(dataset_name: str, dataset: HFDataset, results: dict, laion, k: int = k, sim_threshold: float = 0.88):
    """
    Visualize duplicate pairs of images between target dataset and LAION-400M.
    
    Parameters:
        target_dataset: The dataset to find duplicates for
        laion_dataset: The LAION-400M dataset
        duplicates: List of (target_uid, laion_uid, distance) tuples
        save_dir: Directory to save visualization images
    """
    output_dir = f"data/intermediate/{dataset_name}/plots"
    correct_dir = os.path.join(output_dir, "correct")
    incorrect_dir = os.path.join(output_dir, "incorrect")

    output_indices = f"data/final/{dataset_name}/final_results.json"

    final_results = {}
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(incorrect_dir, exist_ok=True)
    cols = k + 2
    for uid, match_indices in tqdm(results.items(), desc=f"plotting duplicate images for {dataset_name}"):
        fig, axes = plt.subplots(1, cols, figsize=(cols * 3, 3))
        axes[0].text(0.5, 0.5, uid, fontsize=24, ha='center', va='center')
        axes[0].axis("off")

        original_image, original_text, ahash, phash= dataset.get_by_id(uid)
        original_image_resized = resize_image(original_image)
        axes[1].imshow(original_image_resized)
        wrapped_caption = "\n".join(textwrap.wrap(original_text, width=24))
        axes[1].set_title(wrapped_caption)
        axes[1].axis('off')
        orig_input = preprocess(original_image).unsqueeze(0).to(device)
        with torch.no_grad():
            orig_features = model.encode_image(orig_input)
            orig_features /= orig_features.norm(dim=-1, keepdim=True)

        correct = 0
        for j in range (k):
            ax = axes[j + 2]
            if j >= len(match_indices):
                ax.imshow(np.ones((1, 1, 3)))
            else:
                idx = match_indices[j]
                match_image, match_text, _ = laion[idx]
                match_input = preprocess(match_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    match_features = model.encode_image(match_input)
                    match_features /= match_features.norm(dim=-1, keepdim=True)
                similarity = (orig_features @ match_features.T).item()
                
                laion_phash = imagehash.phash(match_image)
                p_dist = abs(phash - laion_phash)
                ax.imshow(match_image)
                caption_match = "dist: " + str(p_dist) + " " + match_text
                wrapped_lines = textwrap.wrap(caption_match, width=24)
                wrapped_caption_match = "\n".join(wrapped_lines[:2])
                ax.set_title(wrapped_caption_match, fontsize=8)
                if similarity >= sim_threshold:
                    correct += 1
                    if uid not in final_results:
                        final_results[uid] = [idx]
                    else:
                        final_results[uid].append(idx)
            ax.axis('off')
        plt.tight_layout()
        if correct > 0:
            plt.savefig(os.path.join(correct_dir, f"{uid}.png"))
        else:
            plt.savefig(os.path.join(incorrect_dir, f"{uid}.png")) # save to another directory
        plt.close(fig)

    with open(output_indices, "w") as f:
        json.dump(final_results, f)
    print(f"Filtered Visualizations saved to {output_dir} and {output_indices}")
    
def classify_caption_gpt(caption, class_name):
    prompt = f"""
        You are a classification system that determines if a caption is relevant to a class name.
        Steps to determine relevance:
        1. Extract key words from the class name and caption.
        2. Expand the class meaning to include:
            its synonyms, hypernyms, hyponyms, inferred words based on category;
            Cause and Effect: e.g., "fire" → "burn, heat, smoke";
            Functional Association: e.g., "key" → "lock, door, security";
            Situational Association: e.g., "beach" → "sand, sunshine, surfing";
            Common Collocations: e.g., "eat" → "rice, breakfast, snacks, chopsticks";
        3. Matching Criteria:
            If the caption contains the exact class name or any expanded synonym and meanings → return "1".
            If the caption has no relation to the class name → return "2".

        Class Name: {class_name}
        Caption: {caption}

        Return only "1" or "2". No explanations.
        """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20
    )
    cleaned_response = response.choices[0].message.content.strip().lower().replace('"', '').replace("'", "")
    return cleaned_response
def process_split(split_name: str, hf_split_dataset):
    """
    Process a single split of the dataset.
    
    Parameters:
        split_name: Name of the split (e.g., "train", "test")
        hf_split_dataset: The HuggingFace dataset split
    """
    print(f"\nProcessing split: {split_name}")
    target_dataset_name = f"{dataset_name}-{split_name}"
    
    # Process target dataset
    target_optimized_dir = f"data/optimized_dataset/{target_dataset_name}"
    if not os.path.exists(os.path.join(target_optimized_dir, "index.json")):
        optimize_hf_to_lightning(hf_split_dataset, target_optimized_dir, image_key=image_key)
    
    # Create datasets
    target_dataset = HFDataset(
        index_file="index.json",
        root_dir=target_optimized_dir,
        lookup=classes if classes else None,
    )
    dataloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # load faiss index
    binary_index_phash = faiss.read_index_binary("lightning_binary_index.bin")
    
    # Find duplicates
    existing_duplicates = f"data/intermediate/{target_dataset_name}/match_indices_{threshold}/combined_results.json"
    if os.path.exists(existing_duplicates):
        duplicates = json.load(open(existing_duplicates, "r"))
    else:
        duplicates = find_duplicates(target_dataset_name, dataloader, threshold, binary_index_phash, k)
 
    # Load LAION-400M dataset
    print("Loading LAION-400M dataset...")
    add_s3_connection("laoin-400m")
    laion = LAOINStreamingDataset(input_dir="/teamspace/s3_connections/laoin-400m")

    # Visualize duplicates
    filter_and_visualize_duplicates(target_dataset_name, target_dataset, duplicates, laion)

def main():
    # Get all available splits
    splits = hf_dataset.keys()
    print(f"Available splits: {splits}")
    
    # Process each split
    for split in splits:
        process_split(split, hf_dataset[split])

if __name__ == "__main__":
    main() 