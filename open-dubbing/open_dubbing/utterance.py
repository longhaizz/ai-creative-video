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

import dataclasses
import hashlib
import json
import os
import shutil
import tempfile

from typing import Any, Dict, Final, List, Tuple

from open_dubbing import logger
from open_dubbing.preprocessing import PreprocessingArtifacts


class Utterance:

    def __init__(self, target_language: str, output_directory: str):
        self.target_language = target_language
        self.output_directory = output_directory

    def _get_file_name(self):
        _UTTERNACE_METADATA_FILE_NAME: Final[str] = "utterance_metadata"

        target_language_suffix = "_" + self.target_language.replace("-", "_").lower()
        utterance_metadata_file = os.path.join(
            self.output_directory,
            _UTTERNACE_METADATA_FILE_NAME + target_language_suffix + ".json",
        )
        return utterance_metadata_file

    def load_utterances(self) -> tuple[Any, PreprocessingArtifacts, Any]:
        utterance_metadata_file = self._get_file_name()

        with open(utterance_metadata_file, "r", encoding="utf-8") as file:
            data = json.load(file)
            utterances = data["utterances"]
            preprocessing_output = PreprocessingArtifacts(
                **data["PreprocessingArtifacts"]
            )
            metadata = data["metadata"]

        return utterances, preprocessing_output, metadata

    def save_utterances(
        self,
        *,
        utterance_metadata: str,
        preprocessing_output: str,
        metadata: Dict[str, str],
        do_hash: bool = True,
        unique_id: bool = True,
    ) -> None:
        """Saves a Python dictionary to a JSON file.

        Returns:
          A path to the saved uttterance metadata.
        """
        utterance_metadata_file = self._get_file_name()

        try:
            all_data = {}
            if unique_id:
                utterance_metadata = self._add_unique_ids(utterance_metadata)

            if do_hash:
                utterance_metadata = self._hash_utterances(utterance_metadata)

            all_data["utterances"] = utterance_metadata
            if preprocessing_output:
                all_data["PreprocessingArtifacts"] = dataclasses.asdict(
                    preprocessing_output
                )
            all_data["metadata"] = metadata

            json_data = json.dumps(all_data, ensure_ascii=False, indent=4)
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, encoding="utf-8"
            ) as temporary_file:

                temporary_file.write(json_data)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            shutil.copy(temporary_file.name, utterance_metadata_file)
            os.remove(temporary_file.name)
            logger().debug(
                "Utterance metadata saved successfully to"
                f" '{utterance_metadata_file}'"
            )
        except Exception as e:
            logger().warning(f"Error saving utterance metadata: {e}")

    def _get_utterance_fields_to_hash(self, utterance):
        filtered_fields = {
            key: value for key, value in utterance.items() if not key.startswith("_")
        }
        return filtered_fields

    def _hash_utterances(self, utterance_metadata):
        for utterance in utterance_metadata:
            filtered_fields = self._get_utterance_fields_to_hash(utterance)
            dict_str = json.dumps(filtered_fields, sort_keys=True)
            _hash = hashlib.sha256(dict_str.encode()).hexdigest()
            utterance["_hash"] = _hash

            for field in ["assigned_voice", "speaker_id"]:
                value = utterance.get(field)
                if value:
                    utterance[f"_{field}_hash"] = hashlib.sha256(
                        value.encode()
                    ).hexdigest()

        return utterance_metadata

    def _add_unique_ids(self, utterance_metadata):
        for idx, utterance in enumerate(utterance_metadata, start=1):
            new_utterance = {"id": idx}
            new_utterance.update(utterance)
            utterance_metadata[idx - 1] = new_utterance

        return utterance_metadata

    def get_files_paths(self, utterance_metadata) -> Tuple[List[str], List[str]]:
        dubbed_paths = []
        paths = []
        for chunk in utterance_metadata:
            if "path" in chunk:
                paths.append(chunk["path"])
            if "dubbed_path" in chunk:
                dubbed_paths.append(chunk["dubbed_path"])

        return paths, dubbed_paths

    def get_modified_utterance_fields(self, utterance):
        modified = []
        for field in utterance:
            field_hash = utterance.get(f"_{field}_hash")
            if not field_hash:
                continue

            field_value = utterance[field]
            current_hash = hashlib.sha256(field_value.encode()).hexdigest()

            if current_hash != field_hash:
                modified.append(field)

        return modified

    def get_modified_utterances(self, utterance_metadata):
        modified = []
        for utterance in utterance_metadata:
            _hash_utterance = utterance["_hash"]
            filtered_fields = self._get_utterance_fields_to_hash(utterance)

            dict_str = json.dumps(filtered_fields, sort_keys=True)
            _hash = hashlib.sha256(dict_str.encode()).hexdigest()
            if _hash_utterance != _hash:
                modified.append(utterance)

        logger().info(f"Modified {len(modified)} utterances")
        return modified

    def get_without_empty_blocks(self, utterance_metadata):
        new_utterance = []

        for utterance in utterance_metadata:
            text = utterance["text"]
            if len(text) == 0:
                logger().debug(f"Removing empty block: {utterance}")
                continue

            new_utterance.append(utterance)

        return new_utterance

    def _get_highest_id(self, utterance_metadata):
        highest_id = 1
        for utterance in utterance_metadata:
            id = utterance["id"]
            if id > highest_id:
                highest_id = id
        return highest_id

    def _create_new_utterance(self, update, new_id):
        mandatory_fields = [
            "speaker_id",
            "translated_text",
            "assigned_voice",
            "gender",
            "start",
            "end",
        ]

        new_utterance = {"id": new_id}

        for field in mandatory_fields:
            value = update.get(field, None)
            if not value:
                logger().warning(
                    f"Missing field '{field}' when adding new utterance with id '{new_id}'"
                )
                return None

            new_utterance[field] = value

        return new_utterance

    def _update_utterance(self, update, utterance):
        updateable_fields = [
            "speaker_id",
            "translated_text",
            "speed",
            "assigned_voice",
            "for_dubbing",
            "gender",
            "start",
            "end",
        ]
        for field in updateable_fields:
            value = update.get(field, None)
            if not value:
                continue

            utterance[field] = value
        return utterance

    def update_utterances(self, utterance_master, utterance_update):
        id_to_update_or_delete = {}
        id_to_create = {}
        utterance_new = []

        for utterance in utterance_update:
            id = utterance["id"]
            operation = utterance["operation"]
            if operation == "create":
                if id == 0:
                    new_id = self._get_highest_id(utterance_master) + 1
                    new_utterance = self._create_new_utterance(utterance, new_id)
                    if new_utterance:
                        utterance_new.append(new_utterance)
                else:
                    id_to_create[id] = utterance
            else:
                id_to_update_or_delete[id] = utterance

        for utterance in utterance_master:
            id = utterance["id"]

            update = id_to_update_or_delete.get(id, None)
            if not update:
                utterance_new.append(utterance)
            else:
                operation = update.get("operation", None)
                if not operation:
                    raise ValueError("No operation field defined")

                if operation == "delete":
                    continue

                if operation != "update":
                    raise ValueError(f"Invalid operation {operation}")

                utterance = self._update_utterance(update, utterance)
                utterance_new.append(utterance)

            create = id_to_create.get(id, None)
            if create:
                new_id = self._get_highest_id(utterance_master) + 1
                new_utterance = self._create_new_utterance(create, new_id)
                if new_utterance:
                    utterance_new.append(new_utterance)

        return utterance_new
