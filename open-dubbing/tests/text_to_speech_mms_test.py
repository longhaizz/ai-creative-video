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

from open_dubbing.text_to_speech_mms import TextToSpeechMMS


class TestTextToSpeechMMS:

    def test_get_available_voices(self):
        api = TextToSpeechMMS()
        voices = api.get_available_voices(language_code="cat")
        assert len(voices) == 1

    def test_get_languages(self):
        api = TextToSpeechMMS()
        languages = api.get_languages()
        assert len(languages) == 1074
