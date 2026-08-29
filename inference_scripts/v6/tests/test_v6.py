import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import warnings

import torch


V6_DIR = Path(__file__).resolve().parents[1]
INFER_TOOLS_DIR = V6_DIR.parent / "infer_tools"
for import_dir in (str(V6_DIR), str(INFER_TOOLS_DIR)):
    if import_dir not in sys.path:
        sys.path.insert(0, import_dir)

from batch_types import TsvInputRow  # noqa: E402
from cosyvoice_generate_from_json_sbatch import (  # noqa: E402
    child_command,
    expand_conversations,
    load_input_conversations,
    resolve_task_count,
    sbatch_prefix,
    write_run_script,
)
from frontend_batch import ZeroShotBatchPreparer  # noqa: E402
from prepare_cosyvoice_tsv_v5 import (  # noqa: E402
    attach_references,
    build_text_file_records,
    load_audio_references,
)
from staged_inference import StagedBatchInferenceRunner  # noqa: E402
from runtime_utils import resolve_batch_sizes, resolve_cpu_threads  # noqa: E402


class FakeFrontend:
    def __init__(self):
        self.normalized = []
        self.tokenized = []

    def text_normalize(self, text, split=True, text_frontend=True):
        self.normalized.append(text)
        return [text] if split else text

    def _extract_text_token(self, text):
        self.tokenized.append(text)
        token_count = max(1, len(text))
        return (
            torch.arange(token_count, dtype=torch.int32).unsqueeze(0),
            torch.tensor([token_count], dtype=torch.int32),
        )

    @staticmethod
    def _extract_speech_feat(_):
        return torch.zeros(1, 4, 80), torch.tensor([4], dtype=torch.int32)

    @staticmethod
    def _extract_speech_token(_):
        return torch.zeros(1, 2, dtype=torch.int32), torch.tensor(
            [2], dtype=torch.int32
        )

    @staticmethod
    def _extract_spk_embedding(_):
        return torch.zeros(1, 192)


class V6PreparationTests(unittest.TestCase):
    def test_audio_path_and_transcript_pairs_stay_aligned(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "refs.jsonl"
            manifest.write_text(
                "\n".join(
                    [
                        json.dumps({"audio_filepath": "Audio/a.wav", "text": "甲"}),
                        json.dumps({"audio_filepath": "Audio/b.wav", "text": "乙"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            references = load_audio_references(
                manifest, "audio_filepath", "text", root
            )
            records = build_text_file_records(["句子一", "句子二"], "text")
            attach_references(records, references, 1, no_shuffle=True)

            self.assertEqual(records[0]["reference_audio_text"], ["甲"])
            self.assertEqual(records[1]["reference_audio_text"], ["乙"])
            self.assertTrue(records[0]["reference_audio_path"][0].endswith("Audio/a.wav"))
            self.assertTrue(records[1]["reference_audio_path"][0].endswith("Audio/b.wav"))

    def test_missing_reference_transcript_fails_with_line_number(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "refs.jsonl"
            manifest.write_text(
                json.dumps({"audio_filepath": "Audio/a.wav", "text": ""}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, r"refs\.jsonl:1"):
                load_audio_references(manifest, "audio_filepath", "text", root)

    def test_json_loader_rejects_unaligned_reference_lists(self):
        with TemporaryDirectory() as temp_dir:
            input_json = Path(temp_dir) / "input.json"
            input_json.write_text(
                json.dumps(
                    [
                        {
                            "id": "x",
                            "text": ["目标"],
                            "reference_audio_path": ["a.wav", "b.wav"],
                            "reference_audio_text": ["甲"],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "same length"):
                load_input_conversations(input_json)

    def test_candidate_expansion_keeps_reference_transcript(self):
        with TemporaryDirectory() as temp_dir:
            input_json = Path(temp_dir) / "input.json"
            input_json.write_text(
                json.dumps(
                    [
                        {
                            "id": "x",
                            "text": ["一", "二"],
                            "reference_audio_path": ["a.wav", "b.wav"],
                            "reference_audio_text": ["甲", "乙"],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            candidates = expand_conversations(load_input_conversations(input_json))
            self.assertEqual(len(candidates), 4)
            self.assertEqual(
                [(item.reference_audio_path, item.reference_audio_text) for item in candidates],
                [("a.wav", "甲"), ("b.wav", "乙"), ("a.wav", "甲"), ("b.wav", "乙")],
            )


class V6FrontendTests(unittest.TestCase):
    @staticmethod
    def _row():
        return TsvInputRow(
            output_index=0,
            row_id="row-1",
            text="目标文本",
            ref_audio_path="reference.wav",
            ref_audio_text="参考转写",
        )

    def test_cosy2_uses_language_token_and_plain_reference_text(self):
        frontend = FakeFrontend()
        preparer = ZeroShotBatchPreparer(
            frontend=frontend,
            sample_rate=24000,
            text_frontend=False,
            model_version="cosy2",
            lang_token="<|zh|>",
            system_prompt="unused",
        )
        prepared = preparer.prepare_rows([self._row()])[0]
        self.assertEqual(prepared.text_for_metadata, "<|zh|>目标文本")
        self.assertIn("参考转写", frontend.tokenized)
        self.assertNotIn("unused参考转写", frontend.tokenized)

    def test_cosy3_uses_system_prompt_without_language_token(self):
        frontend = FakeFrontend()
        system_prompt = "You are a helpful assistant.<|endofprompt|>"
        preparer = ZeroShotBatchPreparer(
            frontend=frontend,
            sample_rate=24000,
            text_frontend=False,
            model_version="cosy3",
            lang_token="",
            system_prompt=system_prompt,
        )
        prepared = preparer.prepare_rows([self._row()])[0]
        self.assertEqual(prepared.text_for_metadata, "目标文本")
        self.assertIn(system_prompt + "参考转写", frontend.tokenized)
        self.assertFalse(prepared.text_for_metadata.startswith("<|"))


class V6FlowTests(unittest.TestCase):
    def test_cosy3_prelookahead_flow_is_upsampled(self):
        class Cosy3Flow:
            token_mel_ratio = 2

            @staticmethod
            def pre_lookahead_layer(value):
                return value + 1

        token_emb = torch.zeros(1, 3, 4)
        h, mel_len = StagedBatchInferenceRunner._encode_flow_tokens(
            Cosy3Flow(),
            token_emb,
            torch.tensor([3], dtype=torch.int32),
            torch.tensor([2], dtype=torch.int32),
            torch.tensor([2], dtype=torch.int32),
        )
        self.assertEqual(tuple(h.shape), (1, 6, 4))
        self.assertEqual(mel_len.tolist(), [6])

    def test_cosy2_encoder_flow_is_retained(self):
        class Cosy2Flow:
            token_mel_ratio = 2

            @staticmethod
            def encoder(value, _length, streaming=False):
                assert streaming is False
                return value.repeat_interleave(2, dim=1), None

        token_emb = torch.zeros(1, 3, 4)
        h, mel_len = StagedBatchInferenceRunner._encode_flow_tokens(
            Cosy2Flow(),
            token_emb,
            torch.tensor([3], dtype=torch.int32),
            torch.tensor([2], dtype=torch.int32),
            torch.tensor([2], dtype=torch.int32),
        )
        self.assertEqual(tuple(h.shape), (1, 6, 4))
        self.assertEqual(mel_len.tolist(), [6])


class V6LlmPrefixTests(unittest.TestCase):
    def test_zero_shot_prefix_includes_prompt_text_and_prompt_speech(self):
        frontend = FakeFrontend()
        preparer = ZeroShotBatchPreparer(
            frontend=frontend,
            sample_rate=24000,
            text_frontend=False,
            model_version="cosy3",
            lang_token="",
            system_prompt="You are a helpful assistant.<|endofprompt|>",
        )
        model_input = preparer.prepare_rows([V6FrontendTests._row()])[0].segment_inputs[
            0
        ].model_input

        hidden_size = 8
        text_embedding = torch.nn.Embedding(512, hidden_size)
        llm_encoder = SimpleNamespace(
            model=SimpleNamespace(
                model=SimpleNamespace(embed_tokens=text_embedding)
            )
        )
        llm_module = SimpleNamespace(
            llm=llm_encoder,
            llm_decoder=torch.nn.Linear(hidden_size, 100),
            llm_embedding=torch.nn.Embedding(2, hidden_size),
            speech_embedding=torch.nn.Embedding(100, hidden_size),
            sos=0,
            task_id=1,
            eos_token=99,
            stop_token_ids=[99],
        )

        class CapturingRunner(StagedBatchInferenceRunner):
            captured_prefixes = None

            def _llm_decode_full_seq(
                self,
                _llm_module,
                prefix_sequences,
                _prefix_lengths,
                _min_len,
                _max_len,
                _max_decode_steps,
                batch_size,
                _stop_token_set,
                _device,
            ):
                self.captured_prefixes = prefix_sequences
                return [[1] for _ in range(batch_size)]

        cosyvoice = SimpleNamespace(
            model=SimpleNamespace(llm=llm_module, device=torch.device("cpu"))
        )
        runner = CapturingRunner(cosyvoice, 1, 1, "raise")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            runner._run_llm_stage_batch([model_input])

        expected_length = (
            1
            + int(model_input["prompt_text_len"].item())
            + int(model_input["text_len"].item())
            + 1
            + int(model_input["llm_prompt_speech_token_len"].item())
        )
        self.assertEqual(runner.captured_prefixes[0].shape[0], expected_length)


class V6ComputeModeTests(unittest.TestCase):
    def test_gpu_and_cpu_batch_defaults(self):
        self.assertEqual(resolve_batch_sizes("gpu", None, None, None), (4, 4, 4))
        self.assertEqual(resolve_batch_sizes("cpu", None, None, None), (1, 1, 1))

    def test_cpu_mode_rejects_batching(self):
        with self.assertRaisesRegex(ValueError, "only supports batch size 1"):
            resolve_batch_sizes("cpu", 2, None, None)
        with self.assertRaisesRegex(ValueError, "only supports batch size 1"):
            resolve_batch_sizes("cpu", 1, 2, 1)

    def test_cpu_threads_prefer_launcher_value_then_slurm(self):
        self.assertEqual(
            resolve_cpu_threads(
                {"COSYVOICE_CPU_THREADS": "4", "SLURM_CPUS_PER_TASK": "2"}
            ),
            4,
        )
        self.assertEqual(resolve_cpu_threads({"SLURM_CPUS_PER_TASK": "2"}), 2)
        self.assertEqual(resolve_cpu_threads({}), 1)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            resolve_cpu_threads({"SLURM_CPUS_PER_TASK": "many"})

    def test_task_count_uses_common_option_and_gpu_alias(self):
        self.assertEqual(
            resolve_task_count(SimpleNamespace(mode="cpu", num_tasks=8, num_gpus=None)),
            8,
        )
        self.assertEqual(
            resolve_task_count(SimpleNamespace(mode="gpu", num_tasks=None, num_gpus=2)),
            2,
        )
        with self.assertRaisesRegex(ValueError, "not both"):
            resolve_task_count(SimpleNamespace(mode="gpu", num_tasks=2, num_gpus=2))
        with self.assertRaisesRegex(ValueError, "only valid"):
            resolve_task_count(SimpleNamespace(mode="cpu", num_tasks=None, num_gpus=2))

    def test_sbatch_defaults_are_mode_specific_and_single_task(self):
        cpu_prefix = sbatch_prefix(SimpleNamespace(mode="cpu", sbatch_cmd=None))
        gpu_prefix = sbatch_prefix(SimpleNamespace(mode="gpu", sbatch_cmd=None))
        self.assertIn("--cpus-per-task=1", cpu_prefix)
        self.assertIn("--mem=10G", cpu_prefix)
        self.assertNotIn("--gres=gpu:1", cpu_prefix)
        self.assertIn("--gres=gpu:1", gpu_prefix)
        self.assertIn("--ntasks=1", cpu_prefix)
        self.assertIn("--ntasks=1", gpu_prefix)

    def test_custom_sbatch_command_is_preserved_and_validated(self):
        args = SimpleNamespace(
            mode="cpu",
            sbatch_cmd=(
                "sbatch --partition=cpu --cpus-per-task=4 --mem=12G --time=01:00:00"
            ),
        )
        prefix = sbatch_prefix(args)
        self.assertIn("--partition=cpu", prefix)
        self.assertIn("--cpus-per-task=4", prefix)
        self.assertIn("--mem=12G", prefix)
        self.assertIn("--wait", prefix)
        self.assertIn("--ntasks=1", prefix)
        with self.assertRaisesRegex(ValueError, "must use --ntasks=1"):
            sbatch_prefix(SimpleNamespace(mode="cpu", sbatch_cmd="sbatch --ntasks=2"))

    def test_cpu_run_script_caps_threads_and_hides_cuda(self):
        with TemporaryDirectory() as temp_dir:
            script_path = Path(temp_dir) / "run.sh"
            write_run_script(
                script_path,
                ["source /tmp/conda.sh"],
                ["python3", "worker.py"],
                "cpu",
            )
            script = script_path.read_text(encoding="utf-8")
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', script)
        self.assertIn('SLURM_CPUS_PER_TASK:-1', script)
        self.assertIn('export OMP_NUM_THREADS="$COSYVOICE_CPU_THREADS"', script)
        self.assertIn('export MKL_NUM_THREADS="$COSYVOICE_CPU_THREADS"', script)

    def test_cpu_child_command_passes_mode_and_unit_batches(self):
        args = SimpleNamespace(
            python_cmd="python3",
            model_path="/tmp/model",
            model_version="cosy3",
            mode="cpu",
            output_tsv_name="generated.tsv",
            failed_tsv_name="failed.tsv",
            batch_size=None,
            llm_batch_size=None,
            flow_batch_size=None,
            min_token_text_ratio=2.0,
            max_token_text_ratio=20.0,
            flow_n_timesteps=10,
            on_error="skip",
            save_workers=1,
            text_frontend=False,
            lang=None,
            system_prompt="You are a helpful assistant.<|endofprompt|>",
            overwrite=False,
        )
        command = child_command(args, Path("input.tsv"), Path("chunk"))
        self.assertEqual(command[command.index("--mode") + 1], "cpu")
        self.assertEqual(command[command.index("--batch_size") + 1], "1")
        self.assertEqual(command[command.index("--llm_batch_size") + 1], "1")
        self.assertEqual(command[command.index("--flow_batch_size") + 1], "1")

if __name__ == "__main__":
    unittest.main()
