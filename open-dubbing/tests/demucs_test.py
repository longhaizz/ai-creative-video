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

import subprocess
import unittest.mock as mock

from unittest.mock import patch

import pytest

from open_dubbing.demucs import Demucs


class TestBuildDemucsCommand:

    @pytest.mark.parametrize(
        "test_name, kwargs, expected_command",
        [
            (
                "basic",
                {},
                (
                    'python -m demucs.separate -o "test" --device cpu --shifts 1'
                    " --overlap 0.25 -j 0 --two-stems vocals --mp3"
                    ' --mp3-bitrate 320 --mp3-preset 2 "audio.mp3"'
                ),
            ),
            (
                "segment",
                {"segment": 60},
                (
                    'python -m demucs.separate -o "test" --device cpu --shifts 1'
                    " --overlap 0.25 -j 0 --two-stems vocals --segment"
                    ' 60 --mp3 --mp3-bitrate 320 --mp3-preset 2 "audio.mp3"'
                ),
            ),
            (
                "no_split",
                {"split": False},
                (
                    'python -m demucs.separate -o "test" --device cpu --shifts 1'
                    " --overlap 0.25 -j 0 --two-stems vocals --no-split"
                    ' --mp3 --mp3-bitrate 320 --mp3-preset 2 "audio.mp3"'
                ),
            ),
        ],
    )
    def test_build_demucs_command(self, test_name, kwargs, expected_command):
        assert (
            Demucs()
            .build_demucs_command(
                audio_file="audio.mp3",
                output_directory="test",
                device="cpu",
                **kwargs,
            )
            .endswith(expected_command)
        )


class TestExtractCommandInfo:

    @pytest.mark.parametrize(
        "test_name, command, expected_folder, expected_ext, expected_input",
        [
            (
                "Standard MP3",
                (
                    "python3 -m demucs.separate -o 'out_folder' --mp3 --two-stems"
                    ' "audio.mp3"'
                ),
                "out_folder",
                ".mp3",
                "audio",
            ),
        ],
    )
    def test_valid_command(
        self, test_name, command, expected_folder, expected_ext, expected_input
    ):
        result = Demucs()._extract_command_info(command)
        assert result == (
            expected_folder,
            expected_ext,
            expected_input,
        ), f"Failed for command: {command}"


class TestAssembleSplitAudioFilePaths:

    @pytest.mark.parametrize(
        "test_name, command, expected_vocals_path, expected_background_path",
        [
            (
                "Standard MP3",
                (
                    'python3 -m demucs.separate -o "test" --device cpu --shifts 1'
                    " --overlap 0.25 --clip_mode rescale -j 0 --two-stems --segment"
                    ' 60 "audio.mp3"'
                ),
                "test/htdemucs/audio/vocals.wav",
                "test/htdemucs/audio/no_vocals.wav",
            ),
        ],
    )
    def test_assemble_paths(
        self, test_name, command, expected_vocals_path, expected_background_path
    ):
        """Tests assembling paths for various Demucs commands and formats."""
        actual_vocals_path, actual_background_path = (
            Demucs().assemble_split_audio_file_paths(command)
        )
        assert [actual_vocals_path, actual_background_path] == [
            expected_vocals_path,
            expected_background_path,
        ], f"Failed for command: {command}"


class TestExecuteDemucsCommand:

    @patch("subprocess.run")
    def test_execute_command_success(self, mock_run):
        mock_run.return_value.stdout = "Command executed successfully"
        mock_run.return_value.stderr = ""
        mock_run.return_value.returncode = 0
        Demucs().execute_demucs_command(command="echo 'Command executed successfully'")
        mock_run.assert_called_once_with(
            "echo 'Command executed successfully'",
            shell=True,
            capture_output=True,
            text=True,
            check=True,
        )

    @mock.patch("subprocess.run")
    def test_execute_command_error(self, mock_run):
        """Test if DemucsCommandError is raised when command fails."""
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "demucs.separate", stderr="Some Demucs error message"
        )

        with pytest.raises(
            Exception,
            match=r"Error in attempt to separate audio.*",
        ):
            Demucs().execute_demucs_command(
                "python3 -m demucs.separate -o out_folder --mp3 --two-stems audio.mp3"
            )
