# Data Artifacts

Large datasets, model caches, feature arrays, and generated indexes are external to Git.

## Production Retrieval Index

| Field | Value |
| --- | --- |
| Filename | `AIC2026_BTC_CLIP32_strict.rvdb` |
| Expected path | `output_keyframes/AIC2026_BTC_CLIP32_strict.rvdb` |
| Size | `2,141,831,655` bytes |
| SHA256 | `54fe6cc5507c51f29c5117601e87b50ae99dbe788ecb0997feba08b90cee594c` |
| Feature source | BTC CLIP features |
| CLIP model | `openai/clip-vit-base-patch32` |
| Embedding dimension | `512` |
| Indexed keyframes | `177,321` |

The checksum was computed read-only from the local production index on 2026-08-27. Verify externally transferred copies before Smart Load.

## Required for Search

- Production `.rvdb` file at any path selectable through Smart Load.
- Access to `openai/clip-vit-base-patch32`, either from the local Hugging Face cache or through an allowed model download.

The production `.rvdb` stores retrieval metadata, including real frame IDs. BTC `.npy` files are not needed for Smart Load or normal search.

## Required for a BTC Index Rebuild

- `clip_features/clip-features-32/{video_id}.npy`
- `map-keyframes-aic25-b1/map-keyframes/{video_id}.csv`
- `Keyframes/{video_id}/{local_keyframe}.jpg`

The strict mapping CSV is the source of truth for converting each local keyframe index to the real video `frame_idx`. A competition build must not fall back to the image filename when mapping fails.

## Optional Dataset Metadata

- `media-info-aic25-b1/` contains external media metadata but is not required for the current Smart Load search path.

## Repository Policy

All paths above are ignored by the root `.gitignore`. Store and distribute these artifacts outside Git. Do not commit generated `.rvdb`, `.npy`, FAISS/index files, keyframes, logs, exports, or model caches.
