import os
import tempfile

import numpy as np
import pytest

from faster_whisper import WhisperModel

from open_dubbing.utterance import Utterance


class TestCmd:

    # TODO:
    #   - To check transcription out of the final video
    def _get_transcription(self, filename):
        model = WhisperModel("medium")
        segments, info = model.transcribe(filename, language="ca", temperature=[0])
        text = ""
        for segment in segments:
            text += segment.text
        return text.strip(), info.language

    def _update_translation(self, directory):
        metadata_file = os.path.join(directory, "utterance_metadata_cat.json")

        with open(metadata_file, "r", encoding="utf-8") as file:
            text = file.read()

        text = text.replace(
            "I m'encanta aquesta ciutat.", "I m'encanta aquesta ciutat tant meva."
        )

        with open(metadata_file, "w", encoding="utf-8") as file:
            file.write(text)

    def _get_utterances(self, directory):
        utterance = Utterance(target_language="cat", output_directory=directory)
        utterances, _, _ = utterance.load_utterances()
        return utterances

    def _assert_dubbing_action(self, directory):
        utterances = self._get_utterances(directory)
        text_array = [entry["translated_text"] for entry in utterances]

        # Common
        assert all(
            "Male" == entry["gender"] for entry in utterances
        ), "Utterance gender check failed"

        starts = [entry["start"] for entry in utterances]
        ends = [entry["end"] for entry in utterances]
        speeds = [entry["speed"] for entry in utterances]

        assert np.allclose(
            [
                1.26284375,
                5.24534375,
                7.607843750000001,
            ],
            starts,
            atol=0.5,
        ), "Utterance start check failed"

        assert np.allclose(
            [
                3.94596875,
                6.629093750000001,
                8.687843750000003,
            ],
            ends,
            atol=0.5,
        ), "Utterance end check failed"

        assert np.allclose(
            [1.0, 1.0, 1.3], speeds, atol=2
        ), "Utterance speed check failed"

        assert "Bon dia, em dic Jordi Mas." == text_array[0], "translated text 0"
        assert "Sóc de Barcelona." == text_array[1], "translated text 1"
        assert "I m'encanta aquesta ciutat." == text_array[2], "translated text 2"

    def _assert_update_action(self, directory):
        utterances = self._get_utterances(directory)
        text_array = [entry["translated_text"] for entry in utterances]

        assert (
            "I m'encanta aquesta ciutat tant meva." == text_array[2]
        ), "updated translated text 2"

    # Cannot do mms due to lack of disk space
    @pytest.mark.parametrize("tts_engine", ["edge"])
    def test_translations_with_tts(self, tts_engine):
        full_path = os.path.realpath(__file__)
        path, _ = os.path.split(full_path)

        _file = os.path.join(path, "englishvideo.mp4")
        dir_obj = tempfile.TemporaryDirectory()
        directory = dir_obj.name

        COMMAND = (
            "open-dubbing "
            f"--input_file={_file} "
            f"--output_directory={directory} "
            "--source_language=eng "
            "--target_language=cat "
            "--nllb_model=nllb-200-1.3B "
            "--whisper_model=medium "
            f"--tts={tts_engine}"
        )
        cmd = f"cd {directory} && {COMMAND}"
        os.system(cmd)

        self._assert_dubbing_action(directory)

        # Update action
        self._update_translation(directory)

        update_command = COMMAND + "--update"
        cmd = f"cd {directory} && {update_command}"
        os.system(cmd)

        self._assert_update_action(directory)
