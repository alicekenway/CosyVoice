import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock


INFER_TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(INFER_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(INFER_TOOLS_DIR))

from process_generated_tts import (  # noqa: E402
    iter_generated_candidates,
    main,
    parse_output_index_range,
)


class ProcessGeneratedTtsTests(unittest.TestCase):
    def test_parse_output_index_range_is_inclusive(self):
        self.assertEqual(parse_output_index_range(None), (0, None))
        self.assertEqual(parse_output_index_range("0:9"), (0, 9))
        self.assertEqual(parse_output_index_range("10:"), (10, None))
        self.assertEqual(parse_output_index_range(":4"), (0, 4))
        self.assertEqual(parse_output_index_range("7"), (7, 7))

    def test_iter_candidates_keeps_all_candidates_in_selected_outputs(self):
        data = [
            {
                "id": "group_1",
                "output": [
                    {"text": "wuw", "candidate_audio_path": ["a.wav"]},
                    {
                        "text": "non wuw",
                        "candidate_audio_path": ["b.wav", "c.wav"],
                    },
                ],
            }
        ]

        candidates = list(
            iter_generated_candidates(data, Path("/audio"), range_start=1, range_end=1)
        )

        self.assertEqual(len(candidates), 2)
        self.assertEqual([candidate.text for candidate in candidates], ["non wuw"] * 2)
        self.assertEqual(
            [candidate.source_audio for candidate in candidates],
            [Path("/audio/b.wav"), Path("/audio/c.wav")],
        )

    def test_main_resamples_and_writes_compatible_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = root / "source"
            source_dir.mkdir()
            for name in ("a.wav", "b.wav"):
                with wave.open(str(source_dir / name), "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(8000)
                    wav_file.writeframes(b"\x00\x00" * 80)

            generated_path = source_dir / "generated.json"
            generated_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "group_1",
                            "output": [
                                {
                                    "text": "Hello Loncin",
                                    "candidate_audio_path": ["a.wav", "b.wav"],
                                }
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            output_dir = root / "prepared"
            argv = [
                "process_generated_tts.py",
                "--input",
                str(generated_path),
                "--output-dir",
                str(output_dir),
                "--sample-rate",
                "16000",
                "--start-index",
                "5",
            ]

            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(main(), 0)

            records = [
                json.loads(line)
                for line in (output_dir / "metadata.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                records,
                [
                    {"audiofile_path": "wav/000000005.wav", "text": "Hello Loncin"},
                    {"audiofile_path": "wav/000000006.wav", "text": "Hello Loncin"},
                ],
            )
            with wave.open(str(output_dir / "wav" / "000000005.wav"), "rb") as wav_file:
                self.assertEqual(wav_file.getframerate(), 16000)


if __name__ == "__main__":
    unittest.main()
