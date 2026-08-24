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

from typing import List

from openai import OpenAI

from open_dubbing import logger
from open_dubbing.text_to_speech import TextToSpeech, Voice


# Documentation: https://platform.openai.com/docs/guides/text-to-speech
class TextToSpeechOpenAI(TextToSpeech):

    def __init__(self, device="cpu", server="", api_key=""):
        super().__init__()
        self.client = OpenAI()
        self.client.api_key = api_key

    def get_available_voices(self, language_code: str) -> List[Voice]:

        voices_names = {
            "alloy": self._SSML_MALE,
            "echo": self._SSML_MALE,
            "fable": self._SSML_MALE,
            "onyx": self._SSML_MALE,
            "nova": self._SSML_FEMALE,
            "shimmer": self._SSML_FEMALE,
        }

        voices = []
        for voice_name in voices_names.keys():
            gender = voices_names[voice_name]

            voice = Voice(
                name=voice_name,
                gender=gender,
                region="",
            )
            voices.append(voice)

        logger().debug(
            f"text_to_speech_openai.get_available_voices: {voices} for language {language_code}"
        )

        return voices

    def _does_voice_supports_speeds(self):
        return False

    def _convert_text_to_speech(
        self,
        *,
        assigned_voice: str,
        target_language: str,
        output_filename: str,
        text: str,
        speed: float,
    ) -> str:

        logger().debug(
            f"text_to_speech_openai._convert_text_to_speech: assigned_voice: {assigned_voice}, output_filename: '{output_filename}'"
        )
        response = self.client.audio.speech.create(
            model="tts-1",
            voice=assigned_voice,
            input=text,
        )

        response.stream_to_file(output_filename)

        return output_filename

    def get_languages(self):
        languages = languages = [
            "afr",
            "ara",
            "hye",
            "aze",
            "bel",
            "bos",
            "bul",
            "cat",
            "zho",
            "hrv",
            "ces",
            "dan",
            "nld",
            "eng",
            "est",
            "fin",
            "fra",
            "glg",
            "deu",
            "ell",
            "heb",
            "hin",
            "hun",
            "isl",
            "ind",
            "ita",
            "jpn",
            "kan",
            "kaz",
            "kor",
            "lav",
            "lit",
            "mkd",
            "msa",
            "mar",
            "mri",
            "nep",
            "nor",
            "fas",
            "pol",
            "por",
            "ron",
            "rus",
            "srp",
            "slk",
            "slv",
            "spa",
            "swa",
            "swe",
            "tgl",
            "tam",
            "tha",
            "tur",
            "ukr",
            "urd",
            "vie",
            "cym",
        ]

        languages = sorted(list(languages))
        logger().debug(f"text_to_speech_openai.get_languages: {languages}")
        return languages
