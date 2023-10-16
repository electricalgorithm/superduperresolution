"""
This file contains the trainer class for the model.
"""
from math import log10
from os import listdir
from os.path import join

from torch import nn, save
from torch.autograd import Variable
from torch.utils.data import DataLoader

from sdr_model.dataset import DatasetsLoader, Dataset
from sdr_model.model import SuperDuperResolution
from sdr_model.video_operations import VideoFile


class Trainer:
    """
    This class handles all the operations related to training the model.
    """

    def __init__(self, model: SuperDuperResolution,
                 criterion: nn.Module,
                 optimizer: nn.Module,
                 train_batch_size: int,
                 test_batch_size: int
                ):
        self.model: nn.Module = model
        self.criterion: nn.Module = criterion
        self.optimizer: nn.Module = optimizer
        self.batch_sizes: dict[str, int] = {
            "train": train_batch_size,
            "test": test_batch_size,
        }

        # Internal attributes.
        self._save_frequency: int = 3
        self._datasets_loader: DatasetsLoader = DatasetsLoader()
        self._video_locations: list[VideoFile] = []
        self._cursor_position: int = 0

    def train_and_test(self, epochs: int) -> None:
        """Trains the model for the given number of epochs.
        Args:
            epochs: number of epochs to train
        """
        for epoch_index in range(1, epochs + 1):
            # Train the model for one epoch.
            self._train_one_epoch(epoch_index)
            self.test()

            # Save the model.
            if epoch_index % self._save_frequency == 0:
                self.save_model(f"model_epoch_{epoch_index}.pth")


    def test(self) -> None:
        """Tests the model.
        """
        total_psnr = 0
        total_dataset_size = 0

        # Load the datasets.
        for test_dataset in self._datasets_loader.test_datasets.values():

            # Create the data loader.
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=self.batch_sizes["test"],
                shuffle=True
            )

            # Test the model.
            for batch in test_dataloader:
                # Get the inputs and targets.
                inputs, target = Variable(batch[0]), Variable(batch[1])

                # Forward pass.
                prediction = self.model(inputs)

                # Calculate the PSNR.
                mean_square_error = self.criterion(prediction, target)
                peak_signal_to_noise_ratio = 10 * log10(1 / mean_square_error.data[0])

                # Update statistics.
                total_psnr += peak_signal_to_noise_ratio
                total_dataset_size += len(test_dataset)
        print(f"===> Avg. PSNR: {(total_psnr / total_dataset_size):.4f} dB")

    def save_model(self, file_path: str) -> None:
        """Saves the model to the given file path.
        Args:
            file_path: path to save the model
        """
        save(self.model.state_dict(), file_path)

    def add_videos(self, videos_dir: str) -> None:
        """Adds videos to the trainer.
        Args:
            videos_dir: directory containing videos
        """
        for video_file in listdir(videos_dir):
            if any(video_file.endswith(extension) for extension in [".mp4", ".hevc"]):
                # Example: Jockey_3840x2160_120fps_420_8bit_HEVC_RAW
                # Split name to extract the frame rate.
                frame_rate_text = video_file.split("_")[2]
                frame_rate = int(frame_rate_text[:frame_rate_text.find("fps")])

                self._video_locations.append(
                    VideoFile(join(videos_dir, video_file), frame_rate)
                )

    def _next_dataset(self) -> Dataset:
        """Gets the next dataset from the loader.
        Returns:
            dataset: Next train dataset
        """
        # Check if there are any more videos to load.
        if self._cursor_position == len(self._video_locations):
            return None

        # Remove the previous dataset from the loader.
        if self._cursor_position > 0:
            self._datasets_loader.delete(self._video_locations[self._cursor_position - 1], "train")

        # Load the next dataset.
        current_video = self._video_locations[self._cursor_position]
        self._cursor_position += 1

        return self._datasets_loader.create(
            video=current_video,
            input_dimensions="864x480",
            output_dimensions="1920x1080",
        )

    def _train_one_epoch(self, epoch_index: int) -> None:
        """Trains the model for one epoch.
        Args:
            epoch_index: epoch index
        """
        epoch_loss = 0
        total_dataset_size = 0

        while True:
            # Load the dataset.
            train_dataset = self._next_dataset()

            # Check if there are any more videos to load.
            if train_dataset is None:
                break

            # Create the data loader.
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.batch_sizes["train"],
                shuffle=True
            )

            # Train the model.
            for iteration, batch in enumerate(train_dataloader, 1):
                # Get the inputs and targets.
                inputs, targets = Variable(batch[0]), Variable(batch[1])
                self.optimizer.zero_grad()

                # Forward pass.
                model_output = self.model(inputs)

                # Calculate the loss.
                loss = self.criterion(model_output, targets)

                # Update statistics.
                epoch_loss += loss.data[0]
                total_dataset_size += len(train_dataset)

                # Backward pass.
                loss.backward()
                self.optimizer.step()

                print(f"===> Epoch[{epoch_index}]({iteration}/{len(train_dataset)}): " \
                    f"Loss: {loss.data[0]:.4f}")
            print(f"Epoch {epoch_index} completed. Loss: {epoch_loss / total_dataset_size}")
