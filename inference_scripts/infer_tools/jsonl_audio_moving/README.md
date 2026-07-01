# JSONL Audio Moving

Copy every `audio_filepath` file from a JSONL manifest into `output_dir/wav`,
then write a new JSONL manifest beside that `wav` directory with updated
`audio_filepath` values.

If copied files have duplicate names, the later files get an index suffix before
the extension, such as `audio.wav`, `audio_1.wav`, `audio_2.wav`.

## Usage

```bash
python3 move_jsonl_audio.py \
  --input input.jsonl \
  --output-dir moved_data
```

This creates:

- `moved_data/wav/`
- `moved_data/input.jsonl`

Use a custom output JSONL filename with `--jsonl-name`:

```bash
python3 move_jsonl_audio.py \
  --input input.jsonl \
  --output-dir moved_data \
  --jsonl-name new_manifest.jsonl
```
