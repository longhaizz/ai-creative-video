# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import subprocess
import sys

from open_dubbing import logger


class Demucs:

    def build_demucs_command(
        self,
        *,
        audio_file: str,
        output_directory: str,
        device: str = "cpu",
        shifts: int = 1,
        overlap: float = 0.25,
        mp3_bitrate: int = 320,
        mp3_preset: int = 2,
        jobs: int = 0,
        split: bool = True,
        segment: int | None = None,
        mp3: bool = True,
    ) -> str:
        """Builds the Demucs audio separation command.

        Args:
            audio_file: The path to the audio file to process.
            output_directory: The output directory for separated tracks.
            device: The device to use ("cuda" or "cpu").
            shifts: The number of random shifts for equivariant stabilization.
            overlap: The overlap between splits.
            mp3_bitrate: The bitrate of converted MP3 files.
            mp3_preset: The encoder preset for MP3 conversion.
            jobs: The number of jobs to run in parallel.
            split: Whether to split audio into chunks.
            segment: The split size for chunks (None for no splitting).
            mp3: Convert output to MP3.

        Returns:
            A string representing the constructed command.

        """
        command_parts = [
            sys.executable,
            "-m",
            "demucs.separate",
            "-o",
            f'"{output_directory}"',
            "--device",
            device,
            "--shifts",
            str(shifts),
            "--overlap",
            str(overlap),
            "-j",
            str(jobs),
            "--two-stems vocals",
        ]
        if not split:
            command_parts.append("--no-split")
        elif segment is not None:
            command_parts.extend(["--segment", str(segment)])
        if mp3:
            command_parts.extend(
                [
                    "--mp3",
                    "--mp3-bitrate",
                    str(mp3_bitrate),
                    "--mp3-preset",
                    str(mp3_preset),
                ]
            )
        command_parts.append(f'"{audio_file}"')
        return " ".join(command_parts)

    def execute_demucs_command(self, command: str) -> None:
        """Executes a Demucs command using subprocess.

        Demucs is a model using AI/ML to detach dialogues
        from the rest of the audio file.

        Args:
            command: The string representing the command to execute.
        """
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, check=True
            )
            logger().debug(result.stdout)
        except subprocess.CalledProcessError as error:
            raise Exception(
                f"Error in attempt to separate audio: {error}\n{error.stderr}"
            )

    def _extract_command_info(self, command: str) -> tuple[str, str, str]:
        """Extracts folder name, output file extension, and input file name (without path) from a Demucs command.

        Args:
            command: The Demucs command string.

        Returns:
            tuple: A tuple containing (folder_name, output_file_extension,
            input_file_name).
        """
        folder_pattern = r"-o\s+(['\"]?)(.+?)\1"
        mp3_pattern = r"--mp3"
        input_file_pattern = r"['\"]?(\w+\.\w+)['\"]?$|\s(\w+\.\w+)$"
        folder_match = re.search(folder_pattern, command)
        mp3_match = re.search(mp3_pattern, command)
        input_file_match = re.search(input_file_pattern, command)
        output_directory = folder_match.group(2) if folder_match else ""
        input_file_name_with_ext = (
            (input_file_match.group(1) or input_file_match.group(2))
            if input_file_match
            else ""
        )
        input_file_name_no_ext = (
            os.path.splitext(input_file_name_with_ext)[0] if input_file_match else ""
        )
        if mp3_match:
            output_file_extension = ".mp3"
        else:
            output_file_extension = ".wav"
        return output_directory, output_file_extension, input_file_name_no_ext

    def assemble_split_audio_file_paths(self, command: str) -> tuple[str, str]:
        """Returns paths to the audio files with vocals and no vocals.

        Returns:
            A tuple with a path to the file with the audio with vocals only
            and the other with the background sound only.
        """
        MODEL_NAME = "htdemucs"
        output_directory, output_file_extension, input_file_name = (
            self._extract_command_info(command)
        )
        audio_vocals_file = os.path.join(
            output_directory,
            MODEL_NAME,
            input_file_name,
            f"vocals{output_file_extension}",
        )
        audio_background_file = os.path.join(
            output_directory,
            MODEL_NAME,
            input_file_name,
            f"no_vocals{output_file_extension}",
        )
        return audio_vocals_file, audio_background_file
