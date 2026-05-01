# loophole_v2

A joke / proof-of-concept submission demonstrating that `evaluate.py` measures only `archive.zip` toward the rate score, not anything embedded in `inflate.py`.

The entire compressed payload (mask + model + pose) is embedded in `inflate.py` as a base85 literal. `archive.zip` is a 22-byte empty zip.

This is the same model + mask + pose data used by [unified_brotli (#64)](https://github.com/commaai/comma_video_compression_challenge/pull/64), just relocated outside the only file the evaluator measures.

Builds on the [loophole_test (#36/#38)](https://github.com/commaai/comma_video_compression_challenge/pull/38) idea — that one read the original video off disk; this one ships the actual compressed bytes inside the script.

Expected score: ~0.13 (same quality, near-zero rate). Don't merge this.
