# JSON Merger

Merge several JSON files into one output JSON file.

## Usage

```bash
python3 merge_json_files.py \
  --inputs file1.json:file2.json:file3.json \
  --output-dir output_dir
```

The default output file is `output_dir/merged.json`. Use `--output-name` to
choose a different file name:

```bash
python3 merge_json_files.py \
  --inputs file1.json:file2.json:file3.json \
  --output-dir output_dir \
  --output-name all_data.json
```

Duplicate input files are allowed and are processed in the order provided:

```bash
python3 merge_json_files.py \
  --inputs file1.json:file1.json \
  --output-dir output_dir
```

## Merge Modes

By default, `--mode auto` chooses behavior from the top-level JSON type:

- If all inputs are arrays, the output is one concatenated array.
- If all inputs are objects, the output is one recursively merged object.
- If the top-level JSON types are mixed, the output is a list where each item is
  one input file's whole JSON value.

You can require a specific mode:

```bash
python3 merge_json_files.py \
  --inputs a.json:b.json \
  --output-dir output_dir \
  --mode array
```

Available modes:

- `auto`: choose `array`, `object`, or `list` automatically.
- `array`: every input must be a top-level array; arrays are concatenated.
- `object`: every input must be a top-level object; later files overwrite
  earlier values for the same key, while nested objects are merged recursively.
- `list`: keep each input file's whole JSON value as one item in the output list.

Use `--indent -1` to write compact JSON instead of pretty-printed JSON.
