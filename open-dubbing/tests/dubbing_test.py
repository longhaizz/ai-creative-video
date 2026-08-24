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
import tempfile

from unittest.mock import patch

import pytest

from open_dubbing import dubbing
from open_dubbing.dubbing import Dubber
from open_dubbing.utterance import Utterance


class TestDubbing:

    @classmethod
    def setup_class(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.temp_dir = cls.temp.name

    @classmethod
    def teardown_class(cls):
        cls.temp.cleanup()

    @pytest.mark.parametrize(
        "original_file, expected_result",
        [
            (
                "Test File With Spaces and Uppercase.mp4",
                "testfilewithspacesanduppercase.mp4",
            ),
            ("Test-File-2024-with-Hyphens.mov", "testfile2024withhyphens.mov"),
            ("lowercasefilename.avi", "lowercasefilename.avi"),
        ],
    )
    def test_rename(self, original_file, expected_result):
        result = dubbing.rename_input_file(original_file)
        assert result == expected_result

    @pytest.mark.parametrize(
        "original_filename, expected_filename",
        [
            (
                "Test File With Spaces and Uppercase.mp4",
                "testfilewithspacesanduppercase.mp4",
            ),
            ("Test-File-2024-with-Hyphens.mov", "testfile2024withhyphens.mov"),
            ("lowercasefilename.avi", "lowercasefilename.avi"),
        ],
    )
    def test_overwrite_input_file(self, original_filename, expected_filename):
        original_full_path = os.path.join(self.temp_dir, original_filename)
        expected_full_path = os.path.join(self.temp_dir, expected_filename)

        with open(original_full_path, "w") as f:
            f.write("Test content")

        dubbing.overwrite_input_file(original_full_path, expected_full_path)
        assert os.path.exists(expected_full_path)

    def _setup_temp_files_for_cleaning(self):
        paths = [os.path.join(self.temp_dir, "test_path_1.mp3")]
        dubbed_paths = [os.path.join(self.temp_dir, "dubbed_path_1.mp3")]
        dubbed_audio_path = os.path.join(self.temp_dir, "dubbed_audio_en.mp3")
        dubbed_vocals_path = os.path.join(self.temp_dir, "dubbed_vocals.mp3")

        for path in paths + dubbed_paths + [dubbed_audio_path, dubbed_vocals_path]:
            with open(path, "w") as f:
                f.write("temp content")

        return paths, dubbed_paths, dubbed_audio_path, dubbed_vocals_path

    def test_run_cleaning_yes(self):
        paths, dubbed_paths, dubbed_audio_path, dubbed_vocals_path = (
            self._setup_temp_files_for_cleaning()
        )

        with patch.object(
            Utterance, "get_files_paths", return_value=(paths, dubbed_paths)
        ):
            obj = Dubber(
                input_file="",
                output_directory=self.temp_dir,
                source_language="",
                target_language="en",
                target_language_region="",
                hugging_face_token="",
                tts=None,
                translation=None,
                stt=None,
                device="cpu",
                device_pyannote=None,
                clean_intermediate_files=True,
            )

            obj.run_cleaning()
            for path in paths + dubbed_paths + [dubbed_audio_path, dubbed_vocals_path]:
                assert not os.path.exists(path), f"File {path} was not deleted"

    def test_run_cleaning_no(self):
        paths, dubbed_paths, dubbed_audio_path, dubbed_vocals_path = (
            self._setup_temp_files_for_cleaning()
        )

        with patch.object(
            Utterance, "get_files_paths", return_value=(paths, dubbed_paths)
        ):
            obj = Dubber(
                input_file="",
                output_directory=self.temp_dir,
                source_language="",
                target_language="en",
                target_language_region="",
                hugging_face_token="",
                tts=None,
                translation=None,
                stt=None,
                device="cpu",
                device_pyannote=None,
                clean_intermediate_files=False,
            )

            obj.run_cleaning()
            for path in paths + dubbed_paths + [dubbed_audio_path, dubbed_vocals_path]:
                assert os.path.exists(path), f"File {path} was deleted"
