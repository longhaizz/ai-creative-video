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

from unittest.mock import patch

import pytest

from open_dubbing.main import (
    _get_openai_key,
    _get_selected_translator,
    _get_selected_tts,
)


class TestMain:

    def test_get_selected_tts_mss(self):
        tts = _get_selected_tts("mms", "", "", "cpu", None)
        assert "TextToSpeechMMS" == type(tts).__name__

    def test_get_selected_tts_edge(self):
        tts = _get_selected_tts("edge", "", "", "cpu", None)
        assert "TextToSpeechEdge" == type(tts).__name__

    def test_get_selected_tts_cli(self):
        directory = os.path.dirname(os.path.realpath(__file__))
        data_json = os.path.join(directory, "data/tts_cli.json")
        tts = _get_selected_tts("cli", data_json, "", "cpu", None)
        assert "TextToSpeechCLI" == type(tts).__name__

    def test_get_selected_tts_cli_no_cfg_file(self):
        with pytest.raises(SystemExit) as excinfo:
            _get_selected_tts("cli", "", "", "cpu", None)

        assert excinfo.type is SystemExit
        assert excinfo.value.code == 108

    def test_get_selected_tts_api(self):
        tts = _get_selected_tts("api", "", "http://tts-server.com", "cpu", None)
        assert "TextToSpeechAPI" == type(tts).__name__

    def test_get_selected_tts_api_no_server(self):
        with pytest.raises(SystemExit) as excinfo:
            _get_selected_tts("api", "", "", "cpu", None)

        assert excinfo.type is SystemExit
        assert excinfo.value.code == 110

    def test_get_selected_translator_apertium(self):
        tts = _get_selected_translator("apertium", "", "apertium_url", "cpu")
        assert "TranslationApertium" == type(tts).__name__

    def test_get_selected_translator_apertium_no_server(self):
        with pytest.raises(SystemExit) as excinfo:
            _get_selected_translator("apertium", "", "", "cpu")

        assert excinfo.type is SystemExit
        assert excinfo.value.code == 109

    def test_get_openai_key_key_is_defined(self):
        provided_key = "test_key_123"
        result = _get_openai_key(key=provided_key)
        assert result == provided_key

    def test_get_openai_key_openai_api_key_defined(self):
        env_key = "env_test_key_456"
        with patch.dict(os.environ, {"OPENAI_API_KEY": env_key}):
            result = _get_openai_key(key=None)
            assert result == env_key

    def test_get_openai_key_no_key_defined(self):
        with pytest.raises(SystemExit) as excinfo, patch.dict(
            os.environ, {"OPENAI_API_KEY": ""}
        ):
            _get_openai_key(key=None)

        assert excinfo.type is SystemExit
        assert excinfo.value.code == 113
