# tongflow-modal-pyscenedetect

Official TongFlow plugin. Shot-boundary detection with **PySceneDetect**, running on [Modal](https://modal.com). No model weights — splits a long video into segments by scene.

## Capabilities

- **Split by shots** (`split-video`) — cut a long video into segments at detected scene boundaries.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |

On first use the plugin deploys to your Modal account automatically and caches the build. No Hugging Face token required.
