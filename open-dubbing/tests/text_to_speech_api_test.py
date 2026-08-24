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

import unittest.mock as mock

from unittest.mock import patch

from open_dubbing.text_to_speech_api import TextToSpeechAPI


class MockResponse:
    def __init__(self, content, status_code=200):
        self.content = content
        self.status_code = status_code


class TestTextToSpeechAPI:
    # Mock response class to simulate requests.get

    # Mock voices data that would be returned by the server
    mock_voices_data = [
        {"id": "voice1", "language": "cat", "gender": "female", "region": "US"},
        {"id": "voice2", "language": "eng", "gender": "male", "region": "UK"},
        {"id": "voice3", "language": "cat", "gender": "female", "region": "ES"},
    ]

    @patch.object(TextToSpeechAPI, "_get_voices", return_value=mock_voices_data)
    def test_get_available_voices(self, mock_get_voices):
        api = TextToSpeechAPI(server="http://mockserver.com")
        voices = api.get_available_voices(language_code="cat")

        assert len(voices) == 2
        assert voices[0].name == "voice1"
        assert voices[1].name == "voice3"

    @patch.object(TextToSpeechAPI, "_get_voices", return_value=mock_voices_data)
    def test_get_languages(self, mock_get_voices):
        api = TextToSpeechAPI(server="http://mockserver.com")
        languages = api.get_languages()

        assert languages == ["cat", "eng"]

    # Path to a real .wav file that exists on your disk for testing
    WAV_FILE_PATH = "/path/to/your/test.wav"

    @mock.patch("requests.get")
    @mock.patch.object(TextToSpeechAPI, "_convert_to_mp3")
    def test_convert_text_to_speech(self, mock_convert_to_mp3, mock_requests_get):
        tts_api = TextToSpeechAPI(server="http://dummyserver.com")

        mock_requests_get.return_value = MockResponse(
            bytes("dummy", "utf-8"), status_code=200
        )

        tts_api._convert_text_to_speech(
            assigned_voice="test_voice",
            target_language="en",
            output_filename="None",
            text="Hello, world!",
            speed=1.0,
        )

        # Ensure requests.get was called with the correct URL
        expected_url = (
            "http://dummyserver.com/speak?voice=test_voice&text=Hello, world!"
        )
        mock_requests_get.assert_called_once_with(expected_url)
        assert mock_convert_to_mp3.called

    @mock.patch("time.sleep", return_value=None)
    @mock.patch("requests.get")
    @mock.patch.object(TextToSpeechAPI, "_convert_to_mp3")
    def test_convert_text_to_speech_with_one_retry(
        self, mock_convert_to_mp3, mock_requests_get, _
    ):
        tts_api = TextToSpeechAPI(server="http://dummyserver.com")

        mock_requests_get.side_effect = [
            MockResponse(bytes("dummy", "utf-8"), status_code=500),
            MockResponse(bytes("dummy", "utf-8"), status_code=200),
        ]

        tts_api._convert_text_to_speech(
            assigned_voice="test_voice",
            target_language="en",
            output_filename="None",
            text="Hello, world!",
            speed=1.0,
        )

        # Ensure requests.get was called twice with the correct URL
        expected_url = (
            "http://dummyserver.com/speak?voice=test_voice&text=Hello, world!"
        )
        assert mock_requests_get.call_count == 2
        mock_requests_get.assert_called_with(expected_url)
        assert mock_convert_to_mp3.called
