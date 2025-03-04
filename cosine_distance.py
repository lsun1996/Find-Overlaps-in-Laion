import torch
from PIL import Image
import open_clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from HF_dataset import HFDataset
import json
import os
from tqdm import tqdm
import logging
from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e31')
model.to(device)
model.eval()

tokenizer = open_clip.get_tokenizer('ViT-B-32')

def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0)
    return image

def encode_text(text):
    with torch.no_grad():
        text_tokens = tokenizer([text]).to(device)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

def encode_image(image):
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features

def cosine_similarity(image_features, text_features):
    return (image_features @ text_features.T).item()

def cosine_distance(image_features, text_features):
    return 1 - cosine_similarity(image_features, text_features)

def process_dataset(dataset_name, lookup_file):
    dataset_dir = f"data/optimized_dataset/{dataset_name}"
    if not os.path.exists(dataset_dir):
        logger.warning(f"Check {dataset_name} file path!")
        return
    index_file = "index.json"
    lookup = json.load(open(lookup_file, "r"))
    dataset = HFDataset(dataset_dir, index_file, lookup)

    """one_copy"""
    one_copy_file = f"filtered_results_all/{dataset_name}-one_copy.json"
    output_one_copy = f"filtered_results_all/distances/one_copy/{dataset_name}-one_copy_distances.json"

    if not os.path.exists(output_one_copy):
        if os.path.exists(one_copy_file):
            one_copy = json.load(open(one_copy_file, "r"))

            distance_one_copy = []
            samples_one_copy = one_copy.keys()
            for key in tqdm(samples_one_copy):
                image, text = dataset.get_by_id(key)
                image = preprocess(image).unsqueeze(0).to(device)
                image_features = encode_image(image)
                text_features = encode_text(text)
                distance = cosine_distance(image_features, text_features)
                distance_one_copy.append(distance)

            with open(output_one_copy, "w") as f:
                json.dump(distance_one_copy, f)
        else:
            logger.warning(f"One_copy file of {dataset_name} not found!")
    else:
        logger.info(f"One copy of {dataset_name} has already been processed, skipping...")

    """multiple_copy"""
    multiple_copy_file = f"filtered_results_all/{dataset_name}-multiple_copy.json"
    output_multiple_copy = f"filtered_results_all/distances/multiple_copy/{dataset_name}-multiple_copy_distances.json"

    if not os.path.exists(output_multiple_copy):
        if os.path.exists(multiple_copy_file):
            multiple_copy = json.load(open(multiple_copy_file, "r"))

            # Compute distances for multiple_copy
            distance_multiple_copy = []
            samples_multiple_copy = multiple_copy.keys()
            for key in samples_multiple_copy:
                image, text = dataset.get_by_id(key)
                image = preprocess(image).unsqueeze(0).to(device)  # Preprocess and move to device
                image_features = encode_image(image)
                text_features = encode_text(text)  # Encode text (corrected)
                distance = cosine_distance(image_features, text_features)
                distance_multiple_copy.append(distance)

            # Save multiple_copy distances
            with open(output_multiple_copy, "w") as f:
                json.dump(distance_multiple_copy, f)
        else:
            logger.warning(f"multiple_copy file of {dataset_name} not found!")
    else: 
        logger.info(f"Multiple copy of {dataset_name} has already been processed, skipping...")
    return

if __name__ == "__main__":
    # dataset_name = "sun397-test"
    # lookup_file = f"data/classes_sun397.json"
    dataset_names = ["imagenet-a", "imagenet-o","imagenet-r", "cifar100", "caltech101", "cars", "aircraft", "country211", "fer2013", "food101", "imagenet-v2", "pets", "sun397"]
    dataset_list = [(dataset_name, f"data/classes_{dataset_name}.json") for dataset_name in dataset_names]
    lookup_files = [f"classes_{name}.json" for name in dataset_list]
    splits = ["train", "test"]

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_dataset, dataset_name+"-"+split, lookup_file)
                for dataset_name, lookup_file in dataset_list
                for split in splits]
    for future in futures:
        future.result()

