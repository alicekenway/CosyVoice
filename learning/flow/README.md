# CosyVoice1 Flow Model, Isolated

This directory extracts the CosyVoice1 flow model path from the upstream `CosyVoice` source tree into a small standalone study package.

Kept:
- `MaskedDiffWithXvec`: the CosyVoice1 token-to-mel flow model
- `ConformerEncoder`: the encoder used by the flow model
- `InterpolateRegulator`: token-length to mel-length expansion
- `ConditionalCFM`: conditional flow matching training and Euler sampling
- `ConditionalDecoder`: the Matcha-based 1D U-Net estimator used inside the flow matcher

Removed on purpose:
- LLM / text generation
- HiFi-GAN / waveform vocoder
- CLI, web UI, dataset pipeline, training harness
- ONNX / TensorRT / runtime serving code
- CosyVoice2/3 causal and DiT variants

Source mapping:
- `CosyVoice/cosyvoice/flow/flow.py` -> `flow/model.py`
- `CosyVoice/cosyvoice/flow/length_regulator.py` -> `flow/length_regulator.py`
- `CosyVoice/cosyvoice/flow/flow_matching.py` -> `flow/flow_matching.py`
- `CosyVoice/cosyvoice/flow/decoder.py` -> `flow/decoder.py`
- `CosyVoice/cosyvoice/transformer/*` -> `flow/encoder.py`
- `CosyVoice/cosyvoice/utils/mask.py`, `common.py` -> `flow/utils.py`

High-level data flow:
1. discrete speech tokens -> embedding
2. Conformer encoder -> contextual token representation
3. length regulator -> expand token-rate features to mel-rate features
4. speaker embedding projection -> speaker conditioning
5. conditional flow matching decoder -> predict mel spectrogram

Expected external dependencies if you want to run this package:
- `torch`
- `einops`
- `matcha`

Notes:
- The code is trimmed for study, not for bit-exact reproduction of every production path.
- The core training objective and inference loop are preserved.
- The Matcha decoder blocks are still imported from the external `matcha` package, because they are not implemented inside CosyVoice itself.
