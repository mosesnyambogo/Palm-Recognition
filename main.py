from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
import mediapipe as mp
import os
import csv
from itertools import combinations
from utils import loadCNNModel, extract_roi, calculate_cosine_similarity

torch.set_default_dtype(torch.float64)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the model
recognitionNetwork = loadCNNModel('weights/resnet18_unfreezed_custom_data.pt', device)
recognitionNetwork.eval()

def extract_features(mp_hands, hands, path: str):

    roi = extract_roi(hands, mp_hands, path)

    # Transform for CNN
    CNNtransformer = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_pil = Image.fromarray(roi)
    img = CNNtransformer(img_pil)
    img = img.view(1,img.size(0),img.size(1),img.size(2))
    img = img.to(device)

    with torch.no_grad():
        resp = recognitionNetwork.forward(img)

    return resp

def compare_folder(mp_hands, hands, folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
    pairs = combinations(image_files, 2)

    results = []
    for image1, image2 in tqdm(pairs):
        features1 = torch.flatten(extract_features(mp_hands, hands, os.path.join(folder_path, image1)))
        features2 = torch.flatten(extract_features(mp_hands, hands, os.path.join(folder_path, image2)))
        similarity = calculate_cosine_similarity(features1, features2)
        results.append([image1, image2, similarity])

    # Save results to CSV
    with open('image_similarity_scores.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image 1', 'Image 2', 'Cosine Similarity'])
        writer.writerows(results)

# Folder path
folder_path = 'test_images'
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

def compare_two_images(mp_hands, hands, image_path1, image_path2, similarity_threshold=0.8):
    features1 = torch.flatten(extract_features(mp_hands, hands, image_path1))
    features2 = torch.flatten(extract_features(mp_hands, hands, image_path2))
    similarity = calculate_cosine_similarity(features1, features2)

    # Compare similarity against threshold
    if similarity >= similarity_threshold:
        result = "Same hand"
    else:
        result = "Different hands"
    return image_path1, image_path2, similarity, result

# Example usage with two specific image paths
image_path1 = 'test_images/00501.tiff'
image_path2 = 'test_images/00505.tiff'
result = compare_two_images(mp_hands, hands, image_path1, image_path2)
print(f"Comparison Result: {result[3]}, Cosine Similarity: {result[2]}")

