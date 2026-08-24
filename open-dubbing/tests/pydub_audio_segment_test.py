# Copyright 2025 Jordi Mas i Hernàndez <jmas@softcatala.org>
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

from open_dubbing.pydub_audio_segment import AudioSegment


class TestPydubAudioSegment:

    def test_from_file(self):
        data_dir = os.path.dirname(os.path.realpath(__file__))
        filename = os.path.join(data_dir, "data/this_is_a_test.mp3")
        audio_segment = AudioSegment.from_file(filename)
        assert 4.493061224489796 == audio_segment.duration_seconds

    def test_from_silent(self):
        DURATION = 2 * 1000
        audio_segment = AudioSegment.silent(duration=DURATION)
        assert 2 == audio_segment.duration_seconds

    def test_normalize_silence_no_change(self):
        silent_seg = AudioSegment.silent(
            duration=1000
        )  # Create a 1-second silent AudioSegment

        normalized_silent = silent_seg.normalize()
        assert normalized_silent.max == 0
        assert normalized_silent == silent_seg

    def test_normalize_silence_changed(self):
        data_dir = os.path.dirname(os.path.realpath(__file__))
        filename = os.path.join(data_dir, "data/this_is_a_test.mp3")
        audio_segment = AudioSegment.from_file(filename)
        normalized = audio_segment.normalize()
        assert audio_segment.max == 32768
        assert normalized.max == 32393
        assert normalized != audio_segment

    def test_get_array_of_samples(self):
        silent_seg = AudioSegment.silent(
            duration=1000
        )  # Create a 1-second silent AudioSegment

        samples = silent_seg.get_array_of_samples()
        assert 11025 == len(samples)
        assert all(sample == 0 for sample in samples)

    def test_get_array_of_len(self):
        silent_seg = AudioSegment.silent(
            duration=1000
        )  # Create a 1-second silent AudioSegment

        samples = len(silent_seg)
        assert 1000 == samples
