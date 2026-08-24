# Copyright 2024 Google LLC
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
import tempfile

from typing import List
from unittest.mock import Mock, patch

import pytest

from open_dubbing.pydub_audio_segment import AudioSegment
from open_dubbing.text_to_speech import TextToSpeech, Voice


class TextToSpeechUT(TextToSpeech):
    def _convert_text_to_speech_without_end_silence(
        self,
        *,
        assigned_voice: str,
        target_language: str,
        output_filename: str,
        text: str,
        speed: float,
    ) -> str:
        pass

    def _convert_text_to_speech(
        self,
        *,
        assigned_voice: str,
        target_language: str,
        output_filename: str,
        text: str,
        speed: float,
    ) -> str:
        pass

    def get_available_voices(self, language_code: str) -> List[Voice]:
        pass

    def get_languages(self):
        pass


class TestTextToSpeech:

    @pytest.mark.parametrize(
        "end_block, expected_result,",
        [
            (60, 1.5),  # Exact division
            (91, 1.0),  # 0.98 -> rounded up (>0.5 change)
            (95, 1.0),  # 0.94 -> rounded up (<0.5 change). Rounded down with round()
        ],
    )
    def test_calculate_target_utterance_speed(self, end_block, expected_result):
        DURATION = 90

        tts = TextToSpeechUT()
        with tempfile.TemporaryDirectory() as tempdir, patch.object(
            tts, "get_start_time_of_next_speech_utterance", return_value=end_block
        ):
            dubbed_audio_mock = AudioSegment.silent(duration=DURATION * 1000)
            dubbed_file_path = os.path.join(tempdir, "dubbed.mp3")
            dubbed_audio_mock.export(dubbed_file_path, format="mp3")
            result = tts._calculate_target_utterance_speed(
                start=0,
                end=end_block,
                dubbed_file=dubbed_file_path,
                utterance_metadata="mocked",
            )
            assert expected_result == result

    def _get_dub_metadata(self):
        return [
            {
                "id": 1,
                "for_dubbing": True,
                "assigned_voice": "en_voice",
                "start": 0,
                "end": 5,
                "translated_text": "Hello world",
                "speed": 1.0,
                "path": "some/path/file.mp3",
                "dubbed_path": "dubbed_file_path",
            },
            {
                "id": 2,
                "for_dubbing": True,
                "assigned_voice": "en_voice",
                "start": 5,
                "end": 10,
                "translated_text": "How are you?",
                "speed": 1.0,
                "path": "some/path/file.mp3",
                "dubbed_path": "dubbed_file_path",
            },
        ].copy()

    @pytest.mark.parametrize(
        "calculated_speed, expect_adjust_called, expected_final_speed",
        [
            (0.5, False, 1.0),  # Speed below 1.0 is 1.0
            (1.2, True, 1.2),  # Speed at 1.2, should adjust speed
            (1.5, True, 1.3),  # Speed exceeds 1.2, clamped to max 1.2
        ],
    )
    def test_dub_utterances_with_speeds(
        self, calculated_speed, expect_adjust_called, expected_final_speed
    ):
        tts = TextToSpeechUT()

        # Mock dependencies
        utterance_metadata = self._get_dub_metadata()
        del utterance_metadata[1]

        output_directory = "/output"
        target_language = "eng"

        with patch.object(
            tts, "_does_voice_supports_speeds", return_value=False
        ), patch.object(
            tts, "_convert_text_to_speech", return_value="dubbed_file_path"
        ), patch.object(
            tts,
            "_convert_text_to_speech_without_end_silence",
            return_value="dubbed_file_path",
        ), patch(
            "open_dubbing.ffmpeg.FFmpeg.adjust_audio_speed"
        ) as mock_adjust_speed, patch.object(
            tts, "_calculate_target_utterance_speed", return_value=calculated_speed
        ):
            result = tts.dub_utterances(
                utterance_metadata=utterance_metadata,
                output_directory=output_directory,
                target_language=target_language,
                audio_file="",
            )

            if expect_adjust_called:
                mock_adjust_speed.assert_called_once()
            else:
                mock_adjust_speed.assert_not_called()

            assert result[0]["speed"] == expected_final_speed

    def test_dub_utterances_modified_no_modification(self):
        tts = TextToSpeechUT()

        utterance_metadata = self._get_dub_metadata()
        output_directory = "/output"
        target_language = "eng"

        with patch.object(
            tts, "_does_voice_supports_speeds", return_value=False
        ), patch.object(
            tts, "_convert_text_to_speech", return_value="dubbed_file_path"
        ), patch.object(
            tts,
            "_convert_text_to_speech_without_end_silence",
            return_value="dubbed_file_path",
        ), patch(
            "open_dubbing.ffmpeg.FFmpeg.adjust_audio_speed"
        ) as mock_adjust_speed, patch.object(
            tts, "_calculate_target_utterance_speed", return_value=1.0
        ):
            result = tts.dub_utterances(
                utterance_metadata=utterance_metadata,
                output_directory=output_directory,
                target_language=target_language,
                audio_file="",
                modified_metadata=[],
            )

            mock_adjust_speed.assert_not_called()
            assert result == utterance_metadata

    def test_dub_utterances_modified_one_modification_increase_speed(self):
        tts = TextToSpeechUT()

        utterance_metadata = self._get_dub_metadata()
        update_metadata = self._get_dub_metadata()
        del update_metadata[0]

        output_directory = "/output"
        target_language = "eng"
        NEW_SPEED = 1.1

        # Mock methods
        with patch.object(
            tts, "_does_voice_supports_speeds", return_value=False
        ), patch.object(
            tts, "_convert_text_to_speech", return_value="dubbed_file_path"
        ), patch.object(
            tts,
            "_convert_text_to_speech_without_end_silence",
            return_value="dubbed_file_path",
        ), patch(
            "open_dubbing.ffmpeg.FFmpeg.adjust_audio_speed"
        ) as mock_adjust_speed, patch.object(
            tts, "_calculate_target_utterance_speed", return_value=NEW_SPEED
        ):
            result = tts.dub_utterances(
                utterance_metadata=utterance_metadata,
                output_directory=output_directory,
                target_language=target_language,
                audio_file="",
                modified_metadata=update_metadata,
            )

            expected_medata = update_metadata.copy()
            expected_medata[0]["speed"] = NEW_SPEED
            mock_adjust_speed.assert_called()

            assert utterance_metadata[0] == result[0]
            assert expected_medata[0] == result[1]

    def test_dub_utterances_modified_one_modification_decrease_speed(self):
        tts = TextToSpeechUT()

        # Mock dependencies
        utterance_metadata = self._get_dub_metadata()
        utterance_metadata[1]["speed"] = "1.2"

        update_metadata = self._get_dub_metadata()
        del update_metadata[0]

        output_directory = "/output"
        target_language = "eng"
        NEW_SPEED = 1.0

        # Mock methods
        with patch.object(
            tts, "_does_voice_supports_speeds", return_value=False
        ), patch.object(
            tts, "_convert_text_to_speech", return_value="dubbed_file_path"
        ), patch.object(
            tts,
            "_convert_text_to_speech_without_end_silence",
            return_value="dubbed_file_path",
        ), patch(
            "open_dubbing.ffmpeg.FFmpeg.adjust_audio_speed"
        ) as mock_adjust_speed, patch.object(
            tts, "_calculate_target_utterance_speed", return_value=NEW_SPEED
        ):
            result = tts.dub_utterances(
                utterance_metadata=utterance_metadata,
                output_directory=output_directory,
                target_language=target_language,
                audio_file="",
                modified_metadata=update_metadata,
            )

            expected_medata = update_metadata.copy()
            expected_medata[0]["speed"] = NEW_SPEED
            mock_adjust_speed.assert_not_called()

            assert utterance_metadata[0] == result[0]
            assert expected_medata[0] == result[1]

    @pytest.mark.parametrize(
        "test_name, utterance_metadata, expected_result",
        [
            (
                "continuous",
                [
                    {
                        "text": "Hello, world!",
                        "start": 1.0,
                        "stop": 2.0,
                        "speaker_id": "speaker1",
                        "for_dubbing": "true",
                        "gender": "male",
                    },
                    {
                        "text": "Hello, world!",
                        "start": 2.0,
                        "stop": 3.0,
                        "speaker_id": "speaker1",
                        "for_dubbing": "true",
                        "gender": "male",
                    },
                ],
                2.0,
            ),
            (
                "uses_empty_space",
                [
                    {
                        "text": "Hello, world!",
                        "start": 1.0,
                        "stop": 2.0,
                        "speaker_id": "speaker1",
                        "for_dubbing": "true",
                        "gender": "male",
                    },
                    {
                        "text": "Hello, world!",
                        "start": 3.0,
                        "stop": 5.0,
                        "speaker_id": "speaker1",
                        "for_dubbing": "true",
                        "gender": "male",
                    },
                ],
                3.0,
            ),
            (
                "read files",
                [
                    {
                        "text": "Hello, world!",
                        "start": 1.0,
                        "stop": 2.0,
                        "speaker_id": "speaker1",
                        "for_dubbing": "true",
                        "gender": "male",
                    },
                ],
                4.0,
            ),
        ],
    )
    def test_get_start_time_of_next_speech_utterance(
        self, test_name, utterance_metadata, expected_result
    ):
        DURATION = 4
        with tempfile.TemporaryDirectory() as tempdir:
            dubbed_audio_mock = AudioSegment.silent(duration=DURATION * 1000)
            dubbed_file_path = os.path.join(tempdir, "dubbed.mp3")
            dubbed_audio_mock.export(dubbed_file_path, format="mp3")

            result = TextToSpeechUT().get_start_time_of_next_speech_utterance(
                utterance_metadata=utterance_metadata,
                start=1.0,
                end=1.5,
                audio_file=dubbed_file_path,
            )
            assert result == expected_result

    def test_get_voices_for_region_only(self):
        voices = [
            Voice(name="Voice1", gender="Male", region="US"),
            Voice(name="Voice2", gender="Female", region="UK"),
            Voice(name="Voice3", gender="Male", region="IN"),
            Voice(name="Voice4", gender="Female", region="IN"),
        ]

        result = TextToSpeechUT().get_voices_for_region_only(
            voices=voices, target_language_region="UK"
        )
        assert 1 == len(result)
        assert "UK" == result[0].region

        result = TextToSpeechUT().get_voices_for_region_only(
            voices=voices, target_language_region="IN"
        )

        assert 2 == len(result)
        assert "IN" == result[0].region
        assert "IN" == result[1].region

        result = TextToSpeechUT().get_voices_for_region_only(
            voices=voices, target_language_region=""
        )
        assert 4 == len(result)
        assert "US" == result[0].region

    @pytest.mark.parametrize(
        "target_language_region, expected_voices",
        [
            ("IN", {1: "Voice3"}),
            ("", {1: "Voice1"}),
        ],
    )
    def test_assign_voices_single_male(self, target_language_region, expected_voices):
        tts = TextToSpeechUT()

        utterance_metadata = [
            {
                "speaker_id": 1,
                "gender": "Male",
            }
        ]

        voices = [
            Voice(name="Voice1", gender="Male", region="US"),
            Voice(name="Voice2", gender="Female", region="UK"),
            Voice(name="Voice3", gender="Male", region="IN"),
            Voice(name="Voice4", gender="Female", region="IN"),
        ]

        tts = TextToSpeechUT()

        with patch.object(tts, "get_available_voices", return_value=voices):
            results = tts.assign_voices(
                utterance_metadata=utterance_metadata,
                target_language="",
                target_language_region=target_language_region,
            )
            assert expected_voices == results

    @pytest.mark.parametrize(
        "target_language_region, expected_voices",
        [
            ("IN", {1: "Voice2"}),
            ("", {1: "Voice1"}),
        ],
    )
    def test_assign_voices_single_male_no_male_voice(
        self, target_language_region, expected_voices
    ):
        tts = TextToSpeechUT()

        utterance_metadata = [
            {
                "speaker_id": 1,
                "gender": "Male",
            }
        ]

        voices = [
            Voice(name="Voice1", gender="Female", region="UK"),
            Voice(name="Voice2", gender="Female", region="IN"),
        ]

        tts = TextToSpeechUT()

        with patch.object(tts, "get_available_voices", return_value=voices):
            results = tts.assign_voices(
                utterance_metadata=utterance_metadata,
                target_language="",
                target_language_region=target_language_region,
            )
            assert expected_voices == results

    @pytest.mark.parametrize(
        "target_language_region, expected_voices",
        [
            ("IN", {1: "Voice3", 2: "Voice3"}),
            ("", {1: "Voice1", 2: "Voice3"}),
        ],
    )
    def test_assign_voices_single_two_males_single_voice(
        self, target_language_region, expected_voices
    ):
        tts = TextToSpeechUT()

        utterance_metadata = [
            {
                "speaker_id": 1,
                "gender": "Male",
            },
            {
                "speaker_id": 2,
                "gender": "Male",
            },
        ]

        voices = [
            Voice(name="Voice1", gender="Male", region="US"),
            Voice(name="Voice2", gender="Female", region="US"),
            Voice(name="Voice3", gender="Male", region="IN"),
            Voice(name="Voice4", gender="Female", region="IN"),
        ]

        tts = TextToSpeechUT()

        with patch.object(tts, "get_available_voices", return_value=voices):
            results = tts.assign_voices(
                utterance_metadata=utterance_metadata,
                target_language="",
                target_language_region=target_language_region,
            )
            assert expected_voices == results

    @pytest.mark.parametrize(
        "target_language_region, expected_voices",
        [
            ("IN", {1: "Voice3", 2: "Voice5"}),
            ("", {1: "Voice1", 2: "Voice3"}),
        ],
    )
    def test_assign_voices_single_two_males_two_voices(
        self, target_language_region, expected_voices
    ):
        tts = TextToSpeechUT()

        utterance_metadata = [
            {
                "speaker_id": 1,
                "gender": "Male",
            },
            {
                "speaker_id": 2,
                "gender": "Male",
            },
        ]

        voices = [
            Voice(name="Voice1", gender="Male", region="US"),
            Voice(name="Voice2", gender="Female", region="UK"),
            Voice(name="Voice3", gender="Male", region="IN"),
            Voice(name="Voice4", gender="Female", region="IN"),
            Voice(name="Voice5", gender="Male", region="IN"),
        ]

        tts = TextToSpeechUT()

        with patch.object(tts, "get_available_voices", return_value=voices):
            results = tts.assign_voices(
                utterance_metadata=utterance_metadata,
                target_language="",
                target_language_region=target_language_region,
            )
            assert expected_voices == results

    def _get_update_utterance_metadata(self):
        return [
            {
                "speaker_id": "2",
                "start": 0,
                "assigned_voice": "Voice0",
                "end": 5,
                "translated_text": "Hello world",
                "speed": 1.0,
                "path": "some/path/file.mp3",
            }
        ]

    def test_update_utterance_metadata_assign_voice_from_speaker(self):
        voices = {"1": "Voice1", "2": "Voice2"}

        utterance_metadata = self._get_update_utterance_metadata()
        tts = TextToSpeechUT()
        updated_utterances = tts.update_utterance_metadata(
            utterance=None,
            utterance_metadata=utterance_metadata,
            assigned_voices=voices,
        )

        assert "Voice2" == updated_utterances[0]["assigned_voice"]

    def test_update_utterance_metadata_assign_voice_from_speaker_id(self):
        voices = {"1": "Voice1", "2": "Voice2"}

        utterance_metadata = self._get_update_utterance_metadata()
        utterance = Mock()
        utterance.get_modified_utterance_fields.return_value = {"speaker_id"}

        tts = TextToSpeechUT()
        updated_utterances = tts.update_utterance_metadata(
            utterance=utterance,
            utterance_metadata=utterance_metadata,
            assigned_voices=voices,
        )

        assert "Voice2" == updated_utterances[0]["assigned_voice"]

    def test_update_utterance_metadata_assign_voice_from_assigned_voice(self):
        voices = {"1": "Voice1", "2": "Voice2"}

        utterance_metadata = self._get_update_utterance_metadata()
        utterance = Mock()
        utterance.get_modified_utterance_fields.return_value = {"assigned_voice"}

        tts = TextToSpeechUT()
        updated_utterances = tts.update_utterance_metadata(
            utterance=utterance,
            utterance_metadata=utterance_metadata,
            assigned_voices=voices,
        )

        assert "Voice0" == updated_utterances[0]["assigned_voice"]
