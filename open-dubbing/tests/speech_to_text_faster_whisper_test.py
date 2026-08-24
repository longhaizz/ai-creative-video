# Copyright 2024 Jordi Mas i Hernàndez <jmas@softcatala.org>
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

from open_dubbing.speech_to_text_faster_whisper import SpeechToTextFasterWhisper


class TestTextToSpeechFasterWhisper:
    @classmethod
    def setup_class(cls):
        """Set up the class by loading the model once."""
        cls.stt = SpeechToTextFasterWhisper()
        cls.stt.load_model()

    def test_transcribe(self):
        data_dir = os.path.dirname(os.path.realpath(__file__))
        filename = os.path.join(data_dir, "data/this_is_a_test.mp3")
        text = self.stt._transcribe(
            vocals_filepath=filename, source_language_iso_639_1="en"
        )
        assert text.strip() == "This is a test."

    def test_detect_language(self):
        data_dir = os.path.dirname(os.path.realpath(__file__))
        filename = os.path.join(data_dir, "data/this_is_a_test.mp3")
        language = self.stt.detect_language(filename)
        assert language == "ell"

    def test_get_languages(self):
        languages = self.stt.get_languages()
        assert len(languages) == 100
        assert "eng" in languages
