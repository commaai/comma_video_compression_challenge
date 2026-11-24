# tcm_ctxmix

`tcm_ctxmix` replaces the token coder for PR #153's 600 unchanged semantic token maps with a different integer coder. The submission adds TCM, a context-mixing C coder, and an accompanying re-coding pass for selected renderer and carrier sections.

The archive decodes the same 600 token maps as #153. In particular, the token stream decodes to SHA-256 `d64c2a094dbdcdf6b381c94810f9d672ac33c3c960eb6ae35ba09df0af51c3a1`. The renderer, carrier and EXT1 grids are #153's data, re-coded losslessly rather than regenerated. The result renders the same frames as #153 while changing the token representation and the compact layout of several non-token fields.

## What this submission adds

The submission adds TCM and its integration. TCM is `tcm/tcm.c`: an integer context-mixing coder and decoder for the token maps, with generated tables in `tcm/tcm1_tables.h` and encoding parameters in `tcm/encoder_params.txt`.

It also adds lossless context-mixing recoding in `tcx_ncm.py`. It re-codes four small pieces of the #153 format: the renderer gaps/fp16 scales and masks, the carrier prefix, the RR5 carrier basis residuals, and the carrier coefficient symbols. `tcx_format.py` packages the compact layout; `tcx_models.py`, `tcx_render.py`, and `inflate.py` join the re-coded sections to the vendored decoder and render path.

The road-plane reprojection formula used by TCM was re-implemented in C from PR #154's formula, rather than copied. `tcm/tcm.c` follows the design of PR #154's `container/mc_context.py composite()`: TCM fits parameters on earlier maps, then applies small per-map corrections stored in the stream. This is a C re-implementation, not copying of #154 source code.

## What comes from earlier work

The core archive data comes from PR #153, `hypery11_carrier_joint`. Its renderer, carrier and 600 token maps are data from that PR and are re-coded losslessly. The renderer and carrier weights and descriptors remain present in the #153 representation or are reconstructed from it. `NOTICE_pr153.txt` is #153's notice verbatim.

The source lineage also includes earlier PRs carried through #153: #152 `size_matters` (Jgolding1998), #151 `darskkaa_rdo` (darskkaa), #150 `semantic_folded` (Sandesh Bharadwaj), #148's HPAC rowpack lineage (Bubu631), #146 `joint_hops` (Ray), #141 `semantic_blocks` (Chase), #140 `semantic_joint_ctxmix` (Alejandro Peña/adpena), #135, #133, and #130 (Fesal Fayed). The vendored decoder, carrier, folding, renderer and semantic files retain their original MIT licensing and provenance.

PR #154, `darskkaa_mc` by darskkaa, contributes the road-plane reprojection formula described above. It also supplied the model for `inflate.sh`'s compiler check and exit-69 pattern; that pattern is also shared with PRs #150 through #153. No #154 source file or source licence is included for these ideas: #154's own folder has no licence file of its own, while the repository licence is MIT.

## Score and size

My own Colab run on a Tesla T4 scored this 139,736-byte archive for 600 samples. The archive has SHA-256:

```text
3b0a8381aafaf4735a7b3c6de1d0bb203a035a090660294c9bf4c81d059d23a1
```

The full-precision score is `0.10923210058565329`; the challenge report prints `0.11`. I also measured #153 and #154 on my T4 runs: #153 scores `0.12478190471791535` and #154 scores `0.12407676008655898`. The submitted archive is 23,353 bytes smaller than #153 and 22,294 bytes smaller than #154.

The original uncompressed input is 37,545,489 bytes. The score is
`100 * segnet + sqrt(10 * posenet) + 25 * rate`.
The measured terms are posenet `1.3279187669468229e-06` and segnet `0.00012543571938294917`. The rate is `139736 / 37545489 = 0.0037217786669391894`.

## T4 time

In the submitted T4 run, full `evaluate.sh` took 661.2 seconds (11 minutes 1.240 seconds). Inflation took 611.1 seconds: renderer and carrier restore 5.1 seconds, even frames 15.6 seconds, TCM wait/decode 554.3 seconds, and odd frames 36.1 seconds. TCM itself accounted for about 575 seconds in that inflate run. A separate T4 solo decode took 570.3–569.8 seconds and peaked around 1.66 GB of memory on one CPU thread. The scoring pass took about 31 seconds. The run used 8 vCPUs, 50 GB RAM, and no swap.

## Decoder-side constants

The decoder supplies format constants and tuned values:

- `tcm/tcm1_tables.h` contains the generated log2, logistic squash and rate tables. The C source has five hash-table-size arrays (`hbs_tab`, `hbs_tab8`, `hbs_tab16`, `hbs_tab16b`, and `hbs_tab32`, totaling 480 bytes), measured from occupancy in the 600 token maps; the shipped stream uses the `hbs=2` table only. It also has tuned reprojection start values `mcp = {44, 348, 261, 0, 0, 0, 0}`.
- `tcx_ncm.py` supplies the 32-byte `GAP_CONSTANTS` structure and walks `GAP_WALK`/`GAP_LENGTHS` for #153's renderer layout.
- `tcx_format.py` rebuilds the fixed RID9 architecture table from tensor sizes and constants, and supplies compact-layout magic, flags, field widths and format names.
- `tcx_format.py` holds SHA-256 integrity digests for the decoded renderer body, carrier body and EXT1 block. `inflate.py` holds the token-map digest. These digests verify output; they do not reconstruct or replace decoded data.
- The TCM stream header contains its format magic, context-off mask and 86 parameter bytes. The encoding parameters are also listed in `tcm/encoder_params.txt`; they are not duplicated by the decoder constants described above.

## Running inflation

The challenge inflater expects an archive directory, an output directory, and a video names file:

```bash
bash submissions/tcm_ctxmix/inflate.sh <archive_dir> <output_dir> <video_names_file>
```

`inflate.sh` compiles `tcm/tcm.c` with `cc` using portable C11 flags and then runs `inflate.py`. It requires a compiler; if `CC` names a compiler that is not found, or compilation fails, it exits with status 69. Set `CC` explicitly if needed:

```bash
CC=cc bash submissions/tcm_ctxmix/inflate.sh <archive_dir> <output_dir> <video_names_file>
```

Inflation uses CUDA in the normal path, matching #153. Set `INFLATE_DEVICE=cpu` only to deliberately use CPU rendering; CPU rendering is slower and is not bit-identical to the CUDA result. `INFLATE_THREADS` controls CPU worker threads in that mode.

## Rebuilding the archive

From the challenge repository root, the compressor is:

```bash
uv run --group cpu bash submissions/tcm_ctxmix/compress.sh [--reference153 <archive.zip>] [--output-dir <dir>] [--device cpu|cuda]
```

Without `--reference153`, the script downloads PR #153's public `archive.zip` and verifies it against SHA-256 `52a98a5f7840648c5b08fe958ac42d14e31f8a1e5452c72995f212a538e144a4`. A supplied reference file must have that digest or the script exits before writing. It decodes #153's token maps with the vendored #153 tools, encodes those maps with this folder's `tcm/tcm.c`, re-codes the non-token sections, and writes `<output-dir>/archive.zip`. The default output directory is this submission folder.

The script needs `cc`, `curl`, and the CPU Python environment with NumPy, Torch and Brotli. It starts from the already-encoded PR #153 archive, not from the source video. It was tested on an Apple M-series Mac with CPU only, taking about 20 minutes; it was not tested on Linux or with `--device cuda`.

It is run as:

```bash
uv run --frozen --group cpu bash submissions/tcm_ctxmix/compress.sh
```

To rebuild from a verified local copy:

```bash
uv run --frozen --group cpu bash submissions/tcm_ctxmix/compress.sh --reference153 /path/to/archive.zip
```

AI agents (Claude) wrote the first-party code and ran the tests under my direction; I made the final decisions and reviewed the first-party code.
