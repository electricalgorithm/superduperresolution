"""
This module contains the dataset class for the project.
"""
from os import listdir, remove
from os.path import join
from torch import Tensor
from torch.utils import data
from torchvision.transforms import Compose, ToTensor
from PIL import Image

from sdr_model.video_operations import FFmpegOperations, VideoFile


class Dataset(data.Dataset):
    """Dataset class for the project"""

    def __init__(self, low_resolution_dir: str, high_resolution_dir: str):
        super(Dataset, self).__init__()

        # Check if the directories contains same number of images.
        if len(listdir(low_resolution_dir)) != len(listdir(high_resolution_dir)):
            raise IndexError(
                "The number of low-resolution images and high-resolution images do not match."
            )

        # Recieve the list of images in the directory.
        self.lr_image_list = [
            join(low_resolution_dir, lr_image_file)
            for lr_image_file in listdir(low_resolution_dir)
            if self.is_image_file(lr_image_file)
        ]
        self.hr_image_list = [
            join(high_resolution_dir, hr_image_file)
            for hr_image_file in listdir(high_resolution_dir)
            if self.is_image_file(hr_image_file)
        ]

    def __getitem__(self, index):
        """Gets the image at the given index.

        Args:
            index (int): index of the image

        Returns:
            (input, output): image pairs at the given index
        """
        lr_image = Image.open(self.lr_image_list[index]).convert("YCbCr")
        hr_image = Image.open(self.hr_image_list[index]).convert("YCbCr")
        return Dataset.to_tensor()(lr_image), Dataset.to_tensor()(hr_image)

    def __len__(self):
        return len(self.lr_image_list)

    @staticmethod
    def is_image_file(filename: str) -> bool:
        """Checks if a file is an image.
        Args:
            filename: name of the file
        Returns:
            bool: True if the filename is an image, False otherwise
        """
        return any(
            filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"]
        )

    @staticmethod
    def to_tensor() -> Tensor:
        """Converts the given image to a tensor.
        Args:
            image: image to be converted
        Returns:
            tensor: converted tensor
        """
        return Compose([ToTensor()])


class DatasetsLoader:
    """This class handles loading of the datasets,
    and provides Dataset instances."""

    def __init__(self) -> None:
        self.train_datasets: dict[str, Dataset] = {}
        self.test_datasets: dict[str, Dataset] = {}

    def create(
        self,
        video: VideoFile,
        input_dimensions: str,
        output_dimensions: str,
    ) -> Dataset:
        """Creates a dataset from the given location.
        Args:
            video: VideoFile instance
            input_dimensions: dimensions of the input, e.g. 854x480
            output_dimensions: dimensions of the output, e.g. 1920x1080
        Returns:
            dataset: Train dataset instance
        """
        # Extract frames from the video.
        FFmpegOperations.scale_and_extract_from_video(
            video.path,
            f".temp/datasets/{video.file_name}/low_resolutions",
            input_dimensions,
            video.frame_rate,
        )
        FFmpegOperations.scale_and_extract_from_video(
            video.path,
            f".temp/datasets/{video.file_name}/high_resolutions",
            output_dimensions,
            video.frame_rate,
        )

        # Create a dataset instance.
        dataset = Dataset(
            f".temp/datasets/{video.file_name}/low_resolutions",
            f".temp/datasets/{video.file_name}/high_resolutions"
        )

        # Split the dataset into train and test.
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = data.random_split(dataset, [train_size, test_size])

        # Add the dataset to the lists.
        self.train_datasets[video.file_name] = train_dataset
        self.test_datasets[video.file_name] = test_dataset

        return train_dataset

    def delete(self, video: VideoFile, delete_type: str = "train") -> None:
        """Deletes the given dataset.
        Args:
            video: VideoFile instance
            delete_type: type of the dataset to be deleted, either train or test
        """
        # Retrieve the dataset from the list.
        train_dataset = self.train_datasets[video.file_name]
        test_dataset = self.test_datasets[video.file_name]

        if delete_type == "train":
            # Remove the images from the disk.
            for lr_image in train_dataset.lr_image_list:
                remove(lr_image)
            for hr_image in train_dataset.hr_image_list:
                remove(hr_image)

            # Remove from map.
            del self.train_datasets[video.file_name]

        elif delete_type == "test":
            # Remove the images from the disk.
            for lr_image in test_dataset.lr_image_list:
                remove(lr_image)
            for hr_image in test_dataset.hr_image_list:
                remove(hr_image)

            # Remove from map.
            del self.test_datasets[video.file_name]
        else:
            raise ValueError("Invalid delete type.")
