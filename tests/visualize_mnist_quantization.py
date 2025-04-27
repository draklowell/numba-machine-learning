import os
import sys
from pathlib import Path
import urllib.request
import gzip
import shutil

import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST

project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from data.core.quantize_cpu import CPUStateDownSampler


def download_mnist_dataset(mnist_path):
    """
    Download MNIST dataset if it doesn't exist

    Args:
        mnist_path: Path to store MNIST data
    """
    # Original MNIST dataset URLs
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = {
        'train-images-idx3-ubyte.gz': 'train-images-idx3-ubyte',
        'train-labels-idx1-ubyte.gz': 'train-labels-idx1-ubyte',
        't10k-images-idx3-ubyte.gz': 't10k-images-idx3-ubyte',
        't10k-labels-idx1-ubyte.gz': 't10k-labels-idx1-ubyte'
    }

    # Mirror URLs if the primary URL doesn't work
    backup_urls = [
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
        "https://ossci-datasets.s3.amazonaws.com/mnist/"
    ]

    os.makedirs(mnist_path, exist_ok=True)

    # Check if files already exist
    all_files_exist = all(os.path.exists(os.path.join(mnist_path, f)) for f in files.values())
    if all_files_exist:
        print("MNIST dataset already downloaded")
        return

    print("Downloading MNIST digits dataset...")
    for gz_file, output_file in files.items():
        output_path = os.path.join(mnist_path, output_file)
        if not os.path.exists(output_path):
            # Try primary URL first
            url = base_url + gz_file
            gz_path = os.path.join(mnist_path, gz_file)

            downloaded = False
            # Try primary URL
            try:
                print(f"Downloading {gz_file}...")
                urllib.request.urlretrieve(url, gz_path)
                downloaded = True
            except Exception as e:
                print(f"Primary URL failed: {e}")

                # Try backup URLs
                for backup_base_url in backup_urls:
                    if downloaded:
                        break

                    try:
                        backup_url = backup_base_url + gz_file
                        print(f"Trying backup URL: {backup_url}")
                        urllib.request.urlretrieve(backup_url, gz_path)
                        downloaded = True
                    except Exception as backup_e:
                        print(f"Backup URL failed: {backup_e}")

            if not downloaded:
                print("All download attempts failed.")
                print("Please manually download the MNIST dataset and place in:", mnist_path)
                return

            # Extract the file
            print(f"Extracting {gz_file}...")
            with gzip.open(gz_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Remove the gzip file
            os.remove(gz_path)

    print("MNIST digits dataset download complete")


def visualize_quantization_effects(mnist_path, bit_widths=None):
    """
    Visualize the effects of different quantization bit widths on MNIST images

    Args:
        mnist_path: Path to MNIST data
        bit_widths: List of bit widths to visualize
    """
    if bit_widths is None:
        bit_widths = [1, 2, 3, 4]

    # Load MNIST data
    mndata = MNIST(mnist_path)
    images, labels = mndata.load_training()

    # Select a few random images
    np.random.seed(42)
    indices = np.random.randint(0, len(images), size=5)
    sample_images = np.array([images[i] for i in indices], dtype=np.uint8).reshape(-1, 28, 28)
    sample_labels = [labels[i] for i in indices]

    # Create figure
    fig, axes = plt.subplots(len(sample_images), len(bit_widths) + 1, figsize=(3 * (len(bit_widths) + 1), 3 * len(sample_images)))

    # For each sample image
    for i, (img, label) in enumerate(zip(sample_images, sample_labels)):
        # Original image
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f"Original (Label: {label})")
        axes[i, 0].axis('off')

        # Quantized versions
        for j, bit_width in enumerate(bit_widths):
            quantizer = CPUStateDownSampler(bit_width)
            quantized_img = quantizer(img.copy())

            # Calculate the number of unique values in the quantized image
            unique_values = len(np.unique(quantized_img))

            axes[i, j + 1].imshow(quantized_img, cmap='gray')
            axes[i, j + 1].set_title(f"{bit_width}-bit ({unique_values} values)")
            axes[i, j + 1].axis('off')

    plt.tight_layout()
    plt.savefig("mnist_quantization_visualization.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # Path to MNIST data folder
    mnist_path = "data/datasets/mnist"

    # Create directory if it doesn't exist
    os.makedirs(mnist_path, exist_ok=True)

    # Download MNIST dataset if needed
    download_mnist_dataset(mnist_path)

    # Visualize different bit widths
    bit_widths = [1, 2, 3, 4]
    visualize_quantization_effects(mnist_path, bit_widths)