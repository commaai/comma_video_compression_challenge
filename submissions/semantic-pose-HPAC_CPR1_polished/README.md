# semantic-pose-HPAC_CPR1_polished

F26 is an `F24S` archive with a fixed-boundary int6 residual table, native RC64
token decoding, and a stored frame-0 selector. The promoted archive is 186,724 bytes and has SHA-256
`12cf5d71a94065184f097c3e40dfe9f1db8402a1a76a80efc76a6956fe1e4004`.

`archive.zip` is intentionally ignored by the challenge repository, while
`report.txt` records the promoted evaluation. The archive is published as a
[hash-pinned F26 release](https://github.com/codexblack/comma_video_compression_challenge/releases/download/semantic-pose-HPAC_CPR1_polished-f26/archive.zip),
which supports `curl -L` and is linked from `PR_BODY.md`.

## Validation

With the promoted archive present in this directory, run:

```bash
python verify_submission.py
```

The verifier checks the archive hash, the single `p` payload, and the fixed
F26 wire format. Inflation requires CUDA. `inflate.sh` compiles the small RC64
decoder into a temporary directory, performs no network access, and writes
`0.raw` through the required challenge interface. The verified reconstruction
is 3,662,409,600 bytes with SHA-256
`cb5b9dd36f2dee33419919f1a705aa4bb41baf986f3a972b83de70a1981dd63c`.

## Rebuild the submitted archive

```bash
bash compress.sh
```

`compress.sh` downloads the hash-pinned promoted archive from the public F26
release, verifies its byte size, SHA-256, and ZIP layout, then writes
`archive.zip`. Use `ARCHIVE_URL=<published-archive-url> bash compress.sh` or
`--archive-url` to provide a mirror, or `--out` to select a different output
path. It intentionally reuses the frozen promoted artifact instead of
packaging the exploratory CUDA search pipeline.

## Decoder scope

This submission contains only the runtime decoder required by F26. The
`cpr1` directory holds the integer model and renderer retained from CPR1;
`runtime` holds the F24S parser and entropy decoders added for F26. Training,
search, archive-building, unused floating-point HPAC, and unused codec paths
remain outside the submission.
