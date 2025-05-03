import gzip
import os
import shutil
import urllib.request

import numpy as np


class Downloader:
    """
    Class for downloading and preparing MNIST dataset.

    Handles downloading, extracting, and validating MNIST data files.
    """

    FILES = {
        "train-images-idx3-ubyte.gz": "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte.gz": "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte.gz": "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte.gz": "t10k-labels-idx1-ubyte",
    }

    def __init__(
        self,
        target_dir: str,
        base_url: str = "https://ossci-datasets.s3.amazonaws.com/mnist/",
        mirror_urls: list[str,] = None,
    ):
        self.target_dir = target_dir
        self.base_url = base_url
        self.mirror_urls = mirror_urls or [
            "http://yann.lecun.com/exdb/mnist/",
            "https://storage.googleapis.com/cvdf-datasets/mnist/",
        ]

        os.makedirs(self.target_dir, exist_ok=True)

    def is_downloaded(self) -> bool:
        """Check if all MNIST files already exist in target directory."""
        return all(
            os.path.exists(os.path.join(self.target_dir, f))
            for f in self.FILES.values()
        )

    def download_file(self, gz_file: str, output_file: str) -> bool:
        output_path = os.path.join(self.target_dir, output_file)
        if os.path.exists(output_path):
            print(f"File already exists: {output_path}")
            return True

        gz_path = os.path.join(self.target_dir, gz_file)
        all_urls = [self.base_url] + self.mirror_urls
        downloaded = False

        for url_base in all_urls:
            if downloaded:
                break

            url = url_base + gz_file
            try:
                print(f"Downloading {url}...")
                urllib.request.urlretrieve(url, gz_path)
                downloaded = True
                print(f"Download successful: {gz_file}")
            except Exception as e:
                print(f"Failed to download from {url}: {e}")

        if not downloaded:
            print(f"Failed to download {gz_file} from all URLs")
            return False

        try:
            print(f"Extracting {gz_file}...")
            with gzip.open(gz_path, "rb") as f_in:
                with open(output_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            os.remove(gz_path)
            return True
        except Exception as e:
            print(f"Failed to extract {gz_file}: {e}")
            return False

    def download_dataset(self) -> bool:
        """
        Download and extract the complete MNIST dataset.

        Returns:
            True if all files were successfully downloaded, False otherwise
        """
        if self.is_downloaded():
            print("MNIST dataset already downloaded")
            return True

        print("Downloading MNIST dataset...")
        success = True

        for gz_file, output_file in self.FILES.items():
            file_success = self.download_file(gz_file, output_file)
            success = success and file_success

        if success:
            print("MNIST dataset download complete")
        else:
            print("MNIST dataset download incomplete")

        return success

    def create_numpy_dataset(
        self, save_path: str | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        import struct

        with open(os.path.join(self.target_dir, "train-images-idx3-ubyte"), "rb") as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            train_images = np.frombuffer(f.read(), dtype=np.uint8)
            train_images = train_images.reshape(-1, rows, cols)

        with open(os.path.join(self.target_dir, "train-labels-idx1-ubyte"), "rb") as f:
            magic, num = struct.unpack(">II", f.read(8))
            train_labels = np.frombuffer(f.read(), dtype=np.uint8)

        with open(os.path.join(self.target_dir, "t10k-images-idx3-ubyte"), "rb") as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            test_images = np.frombuffer(f.read(), dtype=np.uint8)
            test_images = test_images.reshape(-1, rows, cols)

        with open(os.path.join(self.target_dir, "t10k-labels-idx1-ubyte"), "rb") as f:
            magic, num = struct.unpack(">II", f.read(8))
            test_labels = np.frombuffer(f.read(), dtype=np.uint8)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            base_path = os.path.splitext(save_path)[0]
            np.save(f"{base_path}_images.npy", train_images)
            np.save(f"{base_path}_labels.npy", train_labels)
            print(
                f"Saved training data to {base_path}_images.npy and {base_path}_labels.npy"
            )

        return train_images, train_labels, test_images, test_labels


if __name__ == "__main__":

    mnist_dir = "data/datasets/mnist"
    downloader = Downloader(mnist_dir)
    if downloader.download_dataset():
        train_images, train_labels, test_images, test_labels = (
            downloader.create_numpy_dataset(
                save_path="data/datasets/mnist/train_images.npy"
            )
        )
        print(f"Training images shape: {train_images.shape}")
        print(f"Training labels shape: {train_labels.shape}")
        print(f"Test images shape: {test_images.shape}")
        print(f"Test labels shape: {test_labels.shape}")
