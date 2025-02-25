from datasets import load_dataset
import json
import numpy as np
import torch
import open_clip
from open_clip import tokenizer
from PIL import Image
from tqdm import tqdm
from HF_dataset import HFDataset
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import io
import os

"""parameters:"""
dataset_name = "sun397"
# class_type = ""
batch_size = 64
num_workers = 8

# classes = []
classes = json.load(open(f"data/classes_{dataset_name}.json", "r"))
classes = [cls.lower().replace("_", " ") for cls in classes]
"""parameters"""

optimized_dir_train = f"data/optimized_dataset/{dataset_name}-train"
optimized_dir_test = f"data/optimized_dataset/{dataset_name}-test"

# handle the case when the is no class lookup
if not classes:
    classes = set()
    index_file_train = os.path.join(optimized_dir_train, "index.json") if optimized_dir_train else None
    index_file_test = os.path.join(optimized_dir_test, "index.json") if optimized_dir_test else None
    with open(index_file_train, "r") as f:
        data = json.load(f)
        for uid, table in data.items():
            classes.add(table["label"])
            print(table["label"])
            break

if os.path.exists(optimized_dir_train):
    dataset_train = HFDataset(
            index_file = "index.json",
            root_dir=optimized_dir_train,
            lookup=classes if classes else None,
            )
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=num_workers)
else:
    dataset_train = None
    dataloader_train = None

if os.path.exists(optimized_dir_test):
    dataset_test = HFDataset(
            index_file = "index.json",
            root_dir=optimized_dir_test,
            lookup=classes if classes else None,
            )
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
else:
    dataset_test = None
    dataloader_test = None

# open_clip.list_pretrained()
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion400m_e31')
model.to(device)
model.eval()

# encoding text
text_inputs = torch.cat([tokenizer.tokenize(f"a photo of a {c}.") for c in classes]).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_inputs).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)
print("Done encoding text inputs")

# encoding images
def top_predictions(image_input):
  # Calculate features
  with torch.no_grad():
      image_features = model.encode_image(image_input)

  # Pick the top 5 most similar labels for the image
  image_features /= image_features.norm(dim=-1, keepdim=True)

  # Calculate similarity
  similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
  values, indices = similarity[0].topk(5)
  return values, indices
  
# encoding images in batch
def process_batch(images):
    images = torch.stack([preprocess(Image.open(io.BytesIO(img)).convert("RGB")) for img in images]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity.topk(5)
    return values, indices

def evaluate_dataset(dataset_name, dataloader_train, dataloader_test, classes):
    print("\nEvaluating", dataset_name)
    total_images = 0
    top1 = 0
    top5 = 0

    if dataloader_train:
        for batch in tqdm(dataloader_train, desc="Evaluating train set"):
            image_bytes, class_names, _ = batch
            total_images += len(image_bytes)
            value, indices = process_batch(image_bytes)

            for i, class_name in enumerate(class_names):
                predicted_classes = [classes[idx] for idx in indices[i].tolist()]
                if class_name == predicted_classes[0]:
                    top1 += 1
                if class_name in predicted_classes:
                    top5 += 1

    if dataloader_test:
        for batch in tqdm(dataloader_test, desc="Evaluating test set"):
            image_bytes, class_names, _ = batch
            total_images += len(image_bytes)
            value, indices = process_batch(image_bytes)

            for i, class_name in enumerate(class_names):
                predicted_classes = [classes[idx] for idx in indices[i].tolist()]
                if class_name == predicted_classes[0]:
                    top1 += 1
                if class_name in predicted_classes:
                    top5 += 1

    top1_accuracy = top1 / total_images * 100
    top5_accuracy = top5 / total_images * 100

    # print(f"Results for {dataset_name}")
    # print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
    # print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")

    print(f"\n{dataset_name}:", end=" ")
    print(f"Top-1: {top1_accuracy:.2f}%  Top-5: {top5_accuracy:.2f}%")
    return top1_accuracy, top5_accuracy

def evaluate_duplicates(dataset_name, category_name, classes):
    print(f"\nEvaluating duplicates {category_name} in {dataset_name}")
    total_images = 0
    top1 = 0
    top5 = 0
    # train split
    result_dir = f"data/final/{dataset_name}-train/duplicate_categories"
    if os.path.exists(result_dir):
        category = json.load(open(os.path.join(result_dir, f"{category_name}.json"), "r"))
        total_images += len(category)
        for uid in category:
            image, class_name = dataset_train.get_by_id(uid)
            image_input = preprocess(image).unsqueeze(0).to(device)
            
            values, indices = top_predictions(image_input)

            predicted_classes = [classes[idx] for idx in indices.tolist()]

            if class_name == predicted_classes[0]:
                top1 += 1
            if class_name in predicted_classes:
                top5 += 1
        
    # test split
    result_dir = f"data/final/{dataset_name}-test/duplicate_categories"
    if os.path.exists(result_dir):
        category = json.load(open(os.path.join(result_dir, f"{category_name}.json"), "r"))
        total_images += len(category)

        for uid in category:
            image, class_name = dataset_test.get_by_id(uid)
            image_input = preprocess(image).unsqueeze(0).to(device)
            
            values, indices = top_predictions(image_input)

            predicted_classes = [classes[idx] for idx in indices.tolist()]

            if class_name == predicted_classes[0]:
                top1 += 1
            if class_name in predicted_classes:
                top5 += 1

    if total_images == 0:
        print(f"No sample in {dataset_name} - {category_name}")
        return

    top1_accuracy = top1 / total_images * 100
    top5_accuracy = top5 / total_images * 100

    print(f"\n{dataset_name} - {category_name}:", end=" ")
    print(f"Top-1: {top1_accuracy:.2f}%  Top-5: {top5_accuracy:.2f}%")


    return top1_accuracy, top5_accuracy

def evaluate_dataset_without(dataset_name, dataloader_train, dataloader_test, category_name, classes):
    print(f"\nEvaluating {dataset_name} without {category_name}")

    result_dir_train = f"data/final/{dataset_name}-train/duplicate_categories"
    result_dir_test = f"data/final/{dataset_name}-test/duplicate_categories"
    
    # Load the UIDs to exclude for train and test datasets
    category_train = set(json.load(open(os.path.join(result_dir_train, f"{category_name}.json"), "r"))) if os.path.exists(result_dir_train) else None
    category_test = set(json.load(open(os.path.join(result_dir_test, f"{category_name}.json"), "r"))) if os.path.exists(result_dir_test) else None

    total_images = 0
    top1 = 0
    top5 = 0

    def process_batch_without(image_batch, class_batch, uid_batch, category):
        nonlocal total_images, top1, top5
        filtered_indices = [i for i, uid in enumerate(uid_batch) if uid not in category]
        if not filtered_indices:
            return
        filtered_images = [image_batch[i] for i in filtered_indices]
        filtered_classes = [class_batch[i] for i in filtered_indices]

        images = torch.stack([preprocess(Image.open(io.BytesIO(img)).convert("RGB")) for img in filtered_images]).to(device)


        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity.topk(5)
        
        for i, class_name in enumerate(filtered_classes):
            predicted_classes = [classes[idx] for idx in indices[i].tolist()]
            if class_name == predicted_classes[0]:
                top1 += 1
            if class_name in predicted_classes:
                top5 += 1
        total_images += len(filtered_images)
    if dataloader_train:
        for batch in tqdm(dataloader_train, desc="Evaluating train dataset"):
            image_batch, class_batch, uid_batch = batch
            process_batch_without(image_batch, class_batch, uid_batch, category_train)

    if dataloader_test:
        for batch in tqdm(dataloader_test, desc="Evaluating test dataset"):
            image_batch, class_batch, uid_batch = batch
            process_batch_without(image_batch, class_batch, uid_batch, category_test)

    top1_accuracy = top1 / total_images * 100
    top5_accuracy = top5 / total_images * 100

    # print(f"Results for {dataset_name} without {category_name}")
    # print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
    # print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")

    print(f"\n{dataset_name} without {category_name}:", end=" ")
    print(f"Top-1: {top1_accuracy:.2f}%  Top-5: {top5_accuracy:.2f}%")
    return top1_accuracy, top5_accuracy

# evaluate_dataset(dataset_name, dataloader_train, dataloader_test, classes)
evaluate_dataset_without(dataset_name, dataloader_train, dataloader_test, "all_captions", classes)

evaluate_duplicates(dataset_name, "all_captions", classes)
evaluate_duplicates(dataset_name, "correct_captions", classes)
evaluate_duplicates(dataset_name, "relevant_captions", classes)
evaluate_duplicates(dataset_name, "irrelevant_captions", classes)

evaluate_duplicates(dataset_name, "only_correct", classes)
evaluate_duplicates(dataset_name, "only_relevant", classes)
evaluate_duplicates(dataset_name, "only_irrelevant", classes)
evaluate_duplicates(dataset_name, "mixed", classes)
evaluate_duplicates(dataset_name, "correct_and_relevant", classes)
evaluate_duplicates(dataset_name, "correct_and_irrelevant", classes)
evaluate_duplicates(dataset_name, "relevant_and_irrelevant", classes)
