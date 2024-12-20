import sensingsp as ssp
import os
import requests
import json
import random
import zipfile

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
        
def download_zipfile_extract_remove(url,zfile,save_path):
    os.makedirs(save_path, exist_ok=True)

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
        
        return save_path
    else:
        print(f"Failed to download {url}. HTTP {response.status_code}")
        return None

