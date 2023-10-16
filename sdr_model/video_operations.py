"""
This module provides the functionality of handling video operations.
"""
import subprocess
from dataclasses import dataclass


@dataclass
class VideoFile:
    """This class represents the location of a video."""
    path: str
    frame_rate: int

    @property
    def file_name(self) -> str:
        """Returns the name of the video file."""
        return self.path.split("/")[-1].split(".")[0]

    @property
    def file_extension(self) -> str:
        """Returns the extension of the video file."""
        return self.path.split(".")[-1]


class FFmpegOperations:
    """This class handles video operations using ffmpeg tool."""
    ENABLE_DEBUG = False

    # Internal class attributes
    _log_channel = subprocess.DEVNULL if not ENABLE_DEBUG else subprocess.STDOUT

    @staticmethod
    def construct_command(*args) -> list[str]:
        """Constructs a command for ffmpeg.
        Args:
            *args: arguments for the ffmpeg command
        Returns:
            list[str]: list of arguments for ffmpeg
        """
        return ["ffmpeg", "-hide_banner", "-y", *args]

    @staticmethod
    def extract_frames_from_video(
        input_video: str, output_directory: str, frame_rate: int = 30
    ):
        """Extract frames from a video using ffmpeg.
        :param input_video: path to the input video
        :param output_directory: path to the output directory
        :param frame_rate: frame rate of the output frames
        """
        cmd = FFmpegOperations.construct_command(
            "-i",
            input_video,
            "-vf",
            f"fps={frame_rate}",
            f"{output_directory}/output_frames_%04d.png",
        )

        try:
            subprocess.run(cmd, check=True, stderr=FFmpegOperations._log_channel)
            print(
                f"Frames extracted from {input_video} and saved to {output_directory}"
            )
        except subprocess.CalledProcessError as error:
            print("Frame extraction failed.")
            raise error

    @staticmethod
    def scale_and_extract_from_video(
        input_video: str, output_directory: str, dimensions: str, frame_rate: int = 30
    ):
        """Extract frames from a video using ffmpeg.
        :param input_video: path to the input video
        :param output_directory: path to the output directory
        :param dimensions: dimensions of the output frames, e.g. 1280x720
        :param frame_rate: frame rate of the output frames
        """
        ffmpeg_dimensions = f"{dimensions.split('x')[0]}:{dimensions.split('x')[1]}"
        input_video_name = input_video.split("/")[-1].split(".")[0]

        # Create directories if they don't exist.
        subprocess.run(["mkdir", "-p", output_directory], check=True)

        # Extract frames from the video.
        cmd = FFmpegOperations.construct_command(
            "-hwaccel",
            "videotoolbox",
            "-i",
            input_video,
            "-filter_complex",
            f'[0:v]scale={ffmpeg_dimensions}[sc]; [sc]fps={frame_rate}[output]',
            "-map",
            '[output]',
            f"{output_directory}/{input_video_name}_{dimensions}_%04d.png",
        )

        try:
            subprocess.run(cmd, check=True, stderr=FFmpegOperations._log_channel)
            print(
                f"Scaled ({dimensions}) frames extracted from "
                f"{input_video} and saved to {output_directory}"
            )
        except subprocess.CalledProcessError as error:
            print("Scaled frame extraction failed.")
            raise error
