import sensingsp as ssp
import os
import requests
import json
import random
import zipfile
from datetime import datetime, timedelta

def download_file(file_path, save_path):
    """Downloads a file from GitHub and saves it locally."""
    url = f"{ssp.config.hub_REPO}/{file_path}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded: {save_path}")
        return save_path
    else:
        print(f"Failed to download {url}. HTTP {response.status_code}")
        return None


def load_metadata(cache_file="metadata.json"):
    """
    Fetches the metadata.json file from the repository.
    If the file exists locally, returns the local version.
    """
    cache_path = os.path.join(ssp.config.temp_folder, cache_file)
    
    # Check if the metadata file exists locally
    if os.path.exists(cache_path):
        last_modified_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        current_time = datetime.now()

        # Check if the file is more than 2 days old
        if current_time - last_modified_time < timedelta(hours=1):    
            print(f"Loading metadata from local cache: {cache_path}")
            with open(cache_path, "r") as file:
                return json.load(file)

    # Fetch from remote if no local cache exists
    metadata_url = f"{ssp.config.hub_REPO}/{cache_file}"
    print(f"Fetching metadata from {metadata_url}...")
    response = requests.get(metadata_url)
    if response.status_code == 200:
        os.makedirs(ssp.config.temp_folder, exist_ok=True)
        with open(cache_path, "w") as file:
            file.write(response.text)
        return response.json()
    else:
        raise Exception(f"Failed to fetch metadata. HTTP {response.status_code}")


def fetch_file(category, name):
    """
    Fetches a .blend file by category and name based on metadata.json.
    If the category folder doesn't exist in ssp.config.temp_folder, it creates it.
    If the file doesn't exist in the category folder, it downloads and saves it.
    """
    category_folder = os.path.join(ssp.config.temp_folder, "hub")
    os.makedirs(category_folder, exist_ok=True)
    category_folder = os.path.join(category_folder, category)
    os.makedirs(category_folder, exist_ok=True)
    local_file_path = os.path.join(category_folder, f"{name}.blend")
    if os.path.exists(local_file_path):
        print(f"File '{name}.blend' already exists in '{category_folder}'.")
        return local_file_path
    metadata = load_metadata()
    if category not in metadata:
        print(f"Category '{category}' not found in metadata.")
        return None

    for item in metadata[category]:
        if item["name"].lower() == name.lower():
            file_path = item["path"]
            print(f"File '{name}.blend' not found in '{category_folder}'. Downloading...")
            return download_file(file_path, local_file_path)
    print(f"File '{name}' not found in category '{category}'.")
    return None

def visualize_file(category, name):
    category_folder = os.path.join(ssp.config.temp_folder, "hub", category)
    local_file_path = os.path.join(category_folder, f"{name}.blend")
    if not os.path.exists(local_file_path):
        print(f"File '{name}.blend' dosn't exists '{category_folder}'. First fetch it.")
        return
    ssp.utils.open_Blend(local_file_path)
    print(f"Rendering {name}") 
    Triangles = ssp.utils.exportBlenderTriangles()
    image = ssp.utils.renderBlenderTriangles(Triangles)
    ssp.utils.showTileImages([[image,category,name]])
def visualize_hub():
    fetch_all_files()
    visualize_downloaded_files()
def visualize_downloaded_files():
    category_folder = os.path.join(ssp.config.temp_folder, "hub")
    all_images = []
    for root, _, files in os.walk(category_folder):
        for file in files:
            if not file.endswith(".blend"):
                continue
            ssp.utils.open_Blend(os.path.join(root, file))
            print(f"Rendering {file}") 
            Triangles = ssp.utils.exportBlenderTriangles()
            image = ssp.utils.renderBlenderTriangles(Triangles)
            all_images.append([image,os.path.basename(root),file.split(".")[0]])   
    ssp.utils.showTileImages(all_images)        
def list_downloaded_files():
    category_folder = os.path.join(ssp.config.temp_folder, "hub")
    for root, _, files in os.walk(category_folder):
        for file in files:
            # print(os.path.relpath(os.path.join(root, file), category_folder))
            print(os.path.join(root, file))

def fetch_all_files():
    """
    Fetches all .blend files from all categories based on metadata.json.
    If the category folder doesn't exist in ssp.config.temp_folder, it creates it.
    If the file doesn't exist in the category folder, it downloads and saves it.
    """
    metadata = load_metadata()
    if not metadata:
        print("No metadata available.")
        return

    for category, files in metadata.items():
        category_folder = os.path.join(ssp.config.temp_folder, "hub")
        os.makedirs(category_folder, exist_ok=True)
        category_folder = os.path.join(category_folder, category)
        os.makedirs(category_folder, exist_ok=True)

        for item in files:
            name = item["name"]
            local_file_path = os.path.join(category_folder, f"{name}.blend")
            if os.path.exists(local_file_path):
                print(f"File '{name}.blend' already exists in '{category_folder}'.")
            else:
                file_path = item["path"]
                print(f"File '{name}.blend' not found in '{category_folder}'. Downloading...")
                download_file(file_path, local_file_path)

    print("All files downloaded.")
def fetch_random_file():
    """
    Fetches a random .blend file from any category based on metadata.json.
    If the file doesn't exist locally, downloads and saves it.
    """
    metadata = load_metadata()

    # Select a random category
    if not metadata:
        print("Metadata is empty. No files to fetch.")
        return None

    category = random.choice(list(metadata.keys()))
    if not metadata[category]:
        print(f"Category '{category}' is empty.")
        return None

    # Select a random file within the category
    item = random.choice(metadata[category])
    file_path = item["path"]
    name = item["name"]
    category_folder = os.path.join(ssp.config.temp_folder, "hub")
    os.makedirs(category_folder, exist_ok=True)
    category_folder = os.path.join(category_folder, category)
    os.makedirs(category_folder, exist_ok=True)
    local_file_path = os.path.join(category_folder, f"{name}.blend")

    # Ensure category folder exists
    
    # Check if file already exists
    if not os.path.exists(local_file_path):
        print(f"Random file '{name}.blend' not found in '{category_folder}'. Downloading...")
        return download_file(file_path, local_file_path)
    else:
        print(f"Random file '{name}.blend' already exists in '{category_folder}'.")
        return local_file_path
    

def available_files():
    """
    Lists all available files in the metadata, grouped by category.
    """
    metadata = load_metadata()
    
    if not metadata:
        print("No metadata available.")
        return

    print("Available files:")
    for category, files in metadata.items():
        print(f"\nCategory: {category}")
        if files:
            for file_info in files:
                print(f"  - Name: {file_info['name']}, Path: {file_info['path']}")
        else:
            print("  (No files available in this category)")
    return metadata
def download_zipfile_extract_remove(url,zfile,save_path,path_append=True):
    os.makedirs(save_path, exist_ok=True)
    if path_append:
        returnFolder = os.path.join(save_path,zfile.split(".")[0])
    else:
        returnFolder = save_path
    if os.path.exists(returnFolder):
        print(f"Folder '{returnFolder}' already exists.")
        return returnFolder
    print(f"Downloading {url+zfile}...")
    response = requests.get(url+zfile, stream=True)
    if response.status_code == 200:
        with open(os.path.join(save_path,zfile), "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded: {os.path.join(save_path,zfile)}")
        # Extract the ZIP file
        print("Extracting the ZIP file...")
        with zipfile.ZipFile(os.path.join(save_path,zfile), "r") as zip_ref:
            zip_ref.extractall(save_path)
        print(f"Data extracted to {save_path}")
        
        # Clean up: Remove the ZIP file
        os.remove(os.path.join(save_path,zfile))
        print("Temporary ZIP file removed.")
        
        return returnFolder
    else:
        print(f"Failed to download {url}. HTTP {response.status_code}")
        return None


def available_models():
    metadata = load_model_metadata()
    
    if not metadata:
        print("No metadata available.")
        return

    print("Available models:")
    for category, models in metadata.items():
        print(f"\nCategory: {category}")
        if models:
            for model_info in models:
                print(f"  - model: {model_info['model']}")
                print(f"  - best loss: {model_info['best loss']}")
                print(f"  - best loss epoch: {model_info['best loss epoch']}")
                print(f"  - Validation dataset split: {model_info['Validation dataset split']}")
                print(f"  - Validation dataset size: {model_info['Validation dataset size']}")
                print(f"  - path: {model_info['path']}")
        else:
            print("  (No models available in this category)")


def load_model_metadata(cache_file="NNmdoels_metadata.json"):
    cache_path = os.path.join(ssp.config.temp_folder, cache_file)
    
    # Check if the metadata file exists locally
    if os.path.exists(cache_path):
        last_modified_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        current_time = datetime.now()

        # Check if the file is more than 2 days old
        if current_time - last_modified_time < timedelta(hours=1):    
            print(f"Loading metadata from local cache: {cache_path}")
            with open(cache_path, "r") as file:
                return json.load(file)

    # Fetch from remote if no local cache exists
    metadata_url = f"{ssp.config.hub_REPO}/{cache_file}"
    print(f"Fetching metadata from {metadata_url}...")
    response = requests.get(metadata_url)
    if response.status_code == 200:
        os.makedirs(ssp.config.temp_folder, exist_ok=True)
        with open(cache_path, "w") as file:
            file.write(response.text)
        return response.json()
    else:
        raise Exception(f"Failed to fetch metadata. HTTP {response.status_code}")

def fetch_pretrained_model(category, name):
    metadata = load_model_metadata()    
    if not metadata:
        print("No metadata available.")
        return
    if category not in metadata:
        print(f"Category '{category}' not found in metadata.")
        return None
    
    for item in metadata[category]:
        if item["name"].lower() == name.lower():
            print(item)
            file_path = item["path"]
            local_file_path = os.path.join(ssp.config.temp_folder, file_path.split("/")[-1])    
            if os.path.exists(local_file_path):
                print(f"Model '{name}' already exists in '{ssp.config.temp_folder}'.")
                return local_file_path
            return download_file(file_path, local_file_path)
    print(f"File '{name}' not found in category '{category}'.")
    return None
