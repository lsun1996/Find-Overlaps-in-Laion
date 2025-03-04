import os
import json
from lightning_cloud.utils.data_connection import add_s3_connection
from laion_streaming_dataset import LAOINStreamingDataset
import imagehash
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

output_dir = "/teamspace/studios/this_studio/filtered_results_all"

add_s3_connection("laoin-400m")
laion = LAOINStreamingDataset(input_dir="/teamspace/s3_connections/laoin-400m")

def deduplicate_and_classify(dataset_name, split, output_dir):
    result_path = f"data/final/{dataset_name}-{split}/final_results.json"

    if not os.path.exists(result_path):
        logger.info(f"{dataset_name} does not have a {split} split, skipping...")
        return

    output_one_copy = os.path.join(output_dir, f"{dataset_name}-{split}-one_copy.json")
    output_multiple_copy = os.path.join(output_dir, f"{dataset_name}-{split}-multiple_copy.json")

    if os.path.exists(output_one_copy) and os.path.exists(output_multiple_copy):
        logger.info(f"Dataset {dataset_name}-{split} has been fully processed, skipping...")
        return
    one_copy = {}
    multiple_copy = {}

    with open(result_path, "r") as f:
        results = json.load(f)

    for uid, indices in tqdm(results.items(), desc=f"Processing {dataset_name}-{split}"):
        if len(indices) == 1:
            one_copy[uid] = indices
            continue
        filtered_indices = []
        seen = set()
        for idx in indices:
            caption = laion[idx][1]
            image = str(imagehash.phash(laion[idx][0]))
            if (caption, image) in seen: # two samples have the same captions and same images are considered duplicates
                continue
            seen.add((caption, image))
            filtered_indices.append(idx)

        if len(filtered_indices) == 1:
            one_copy[uid] = filtered_indices
        else:
            multiple_copy[uid] = filtered_indices

    with open(output_one_copy, "w") as f:
        json.dump(one_copy, f, indent=4)
    with open(output_multiple_copy, "w") as f:
        json.dump(multiple_copy, f, indent=4)

    logger.info(f"Successfully deduplicate and classfied {dataset_name}-{split}")
    return

dataset_list = ["imagenet-a", "imagenet-o","imagenet-r", "cifar100", "caltech101", "cars", "aircraft", "country211", "fer2013", "food101", "imagenet-v2", "pets", "sun397"]
splits = ["train", "test"]
with ProcessPoolExecutor() as executor:
    futures = [executor.submit(deduplicate_and_classify, dataset_name, split, output_dir)
                for dataset_name in dataset_list
                for split in splits]
    for future in futures:
        future.result()
