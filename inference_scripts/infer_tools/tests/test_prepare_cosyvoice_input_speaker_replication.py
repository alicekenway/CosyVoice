import json
import random
import sys
import tempfile
import unittest
from pathlib import Path


INFER_TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(INFER_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(INFER_TOOLS_DIR))

from prepare_cosyvoice_input_speaker_replication import (  # noqa: E402
    build_records,
    load_speaker_audio,
    load_speaker_references,
)


class PrepareSpeakerReplicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.jsonl_path = Path(self.temporary_directory.name) / "references.jsonl"
        records = [
            {
                "speaker": "speaker_a",
                "audio_filepath": "/audio/a.wav",
                "text": "transcript a",
            },
            {
                "speaker": "speaker_a",
                "audio_filepath": "/audio/b.wav",
                "text": "transcript b",
            },
            {
                "speaker": "speaker_a",
                "audio_filepath": "/audio/c.wav",
            },
        ]
        self.jsonl_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_cosy2_preserves_audio_only_output(self):
        references_by_speaker, total_records, skipped_records = (
            load_speaker_references(
                self.jsonl_path,
                "speaker",
                "audio_filepath",
                model_version="cosy2",
            )
        )
        records = build_records(
            texts=["target"],
            audio_by_speaker=references_by_speaker,
            speakers=["speaker_a"],
            replication_index=1,
            candidate_number=3,
            id_prefix="group",
            rng=random.Random(0),
            selection="round-robin",
            model_version="cosy2",
        )

        self.assertEqual(total_records, 3)
        self.assertEqual(skipped_records, 0)
        self.assertEqual(len(records[0]["reference_audio_path"]), 3)
        self.assertNotIn("reference_audio_text", records[0])

    def test_cosy3_writes_aligned_reference_transcripts(self):
        references_by_speaker, total_records, skipped_records = (
            load_speaker_references(
                self.jsonl_path,
                "speaker",
                "audio_filepath",
                model_version="cosy3",
                audio_text_key="text",
            )
        )
        records = build_records(
            texts=["target"],
            audio_by_speaker=references_by_speaker,
            speakers=["speaker_a"],
            replication_index=1,
            candidate_number=2,
            id_prefix="group",
            rng=random.Random(0),
            selection="round-robin",
            model_version="cosy3",
        )

        self.assertEqual(total_records, 3)
        self.assertEqual(skipped_records, 1)
        record = records[0]
        expected_transcript_by_path = {
            "/audio/a.wav": "transcript a",
            "/audio/b.wav": "transcript b",
        }
        self.assertEqual(
            record["reference_audio_text"],
            [
                expected_transcript_by_path[path]
                for path in record["reference_audio_path"]
            ],
        )

    def test_legacy_loader_still_returns_path_strings(self):
        audio_by_speaker, _, _ = load_speaker_audio(
            self.jsonl_path,
            "speaker",
            "audio_filepath",
        )

        self.assertEqual(
            audio_by_speaker["speaker_a"],
            ["/audio/a.wav", "/audio/b.wav", "/audio/c.wav"],
        )
        records = build_records(
            texts=["target"],
            audio_by_speaker=audio_by_speaker,
            speakers=["speaker_a"],
            replication_index=1,
            candidate_number=1,
            id_prefix="group",
            rng=random.Random(0),
            selection="round-robin",
        )
        self.assertIsInstance(records[0]["reference_audio_path"][0], str)


if __name__ == "__main__":
    unittest.main()
