"""
This module contains the dataset class for the project.
"""
from os import listdir, remove
from os.path import join, isdir
import logging
import random

from torch import Tensor
from torch.utils import data
from torchvision.transforms import Compose, ToTensor
from PIL import Image

from sdr_model.video_operations import FFmpegOperations, VideoFile


# Create a logger.
logger = logging.getLogger(__name__)


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
        logger.info("Image lists are retrieved.")

    def __getitem__(self, index):
        """Gets the image at the given index.

        Args:
            index (int): index of the image

        Returns:
            (input, output): image pairs at the given index
        """
        lr_image = Image.open(self.lr_image_list[index]).convert("YCbCr")
        hr_image = Image.open(self.hr_image_list[index]).convert("YCbCr")
        logger.debug("Image pair is retrieved at index %d.", index)

        return Dataset.to_tensor()(lr_image), Dataset.to_tensor()(hr_image)

    def __len__(self):
        return len(self.lr_image_list)

    def shrink(self, new_size: int) -> None:
        """It removes the data in the dataset until the size of the dataset
        becomes equal to the given size.
        """
        # Get a random list of elements without repeated indices, with a length
        # of old-size - new-size.
        indices_to_be_removed = random.sample(
            range(len(self.lr_image_list)), len(self.lr_image_list) - new_size
        )

        # Remove the images from the disk.
        for index in indices_to_be_removed:
            remove(self.lr_image_list[index])
            remove(self.hr_image_list[index])

        # Remove the images from the lists.
        self.lr_image_list = [
            image for index, image in enumerate(self.lr_image_list)
            if index not in indices_to_be_removed
        ]
        self.hr_image_list = [
            image for index, image in enumerate(self.hr_image_list)
            if index not in indices_to_be_removed
        ]

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
        dataset_amount: int,
    ) -> Dataset:
        """Creates a dataset from the given location.
        Args:
            video: VideoFile instance
            input_dimensions: dimensions of the input, e.g. 864x480
            output_dimensions: dimensions of the output, e.g. 1920x1080
        Returns:
            dataset: Train dataset instance
        """
        # Check if frames are already extracted by looking at the directory.
        if not isdir(f".temp/datasets/{video.file_name}/low_resolutions_scaled"):
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
            FFmpegOperations.scale_images(
                f".temp/datasets/{video.file_name}/low_resolutions",
                f".temp/datasets/{video.file_name}/low_resolutions_scaled",
                "png",
                output_dimensions,
                file_name_prefix=f"{video.file_name}_{input_dimensions}_"
            )
            logger.debug("FFmpeg operations completed.")
        else:
            logger.debug("Frames are already extracted. Using them.")

        # Create a dataset instance.
        dataset = Dataset(
            f".temp/datasets/{video.file_name}/low_resolutions_scaled",
            f".temp/datasets/{video.file_name}/high_resolutions"
        )

        # Shrink the dataset.
        dataset.shrink(dataset_amount)

        # Split the dataset into train and test.
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = data.random_split(dataset, [train_size, test_size])

        # Add the dataset to the lists.
        self.train_datasets[video.file_name] = train_dataset
        self.test_datasets[video.file_name] = test_dataset
        logger.debug("Train and test datasets are created.")

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
            logger.debug("Train dataset is deleted.")

        elif delete_type == "test":
            # Remove the images from the disk.
            for lr_image in test_dataset.lr_image_list:
                remove(lr_image)
            for hr_image in test_dataset.hr_image_list:
                remove(hr_image)

            # Remove from map.
            del self.test_datasets[video.file_name]
            logger.debug("Test dataset is deleted.")

        else:
            raise ValueError("Invalid delete type.")
