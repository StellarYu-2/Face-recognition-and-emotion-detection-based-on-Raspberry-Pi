# ASDUN Cloud Server

This is Stage C/D of the hybrid cloud inference plan in `try.md`.

Current goals:

- Start a FastAPI service on the Windows/NVIDIA host.
- Verify ONNX Runtime providers, especially `CUDAExecutionProvider`.
- Accept a face crop through `/analyze`.
- Return fake identity data for Raspberry Pi client integration tests.
- Run real emotion inference with `models/emotion-ferplus-8.onnx`.
- Run cloud identity matching with InsightFace `buffalo_l` when available, or the legacy crop-only ArcFace model.

The preferred identity path is now cloud-side InsightFace detection + alignment + embedding. This is much more reliable
than resizing a Raspberry Pi face crop directly into an ArcFace model.

## Setup Recommended

From the repository root on Windows PowerShell:

```powershell
conda env create -f .\cloud_server\environment.yml
conda activate asdun-cloud
```

If you already created the environment and only want to update dependencies:

```powershell
conda env update -n asdun-cloud -f .\cloud_server\environment.yml --prune
```

Conda is recommended on the Windows/NVIDIA host because later GPU packages are easier to isolate and adjust. The helper script still falls back to `.venv` if conda/mamba is not installed.

## Setup Fallback

If conda is unavailable, the old venv path still works:

```powershell
cd cloud_server
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
pip install -r requirements-gpu.txt
```

For the preferred cloud identity backend, install InsightFace too:

```powershell
pip install -r requirements-insightface.txt
```

Place the complete `buffalo_l` model folder here:

```text
models/buffalo_l/
  det_10g.onnx
  w600k_r50.onnx
  2d106det.onnx
  1k3d68.onnx
  genderage.onnx
```

The code uses `FaceAnalysis(name="buffalo_l", root=".")`, so InsightFace expects the folder at `models/buffalo_l`.

If `onnxruntime-gpu` installation fails temporarily, install CPU runtime just for Stage A plumbing. Stage B should use GPU again:

```powershell
pip install -r requirements-cpu.txt
```

## Run

```powershell
cd cloud_server
conda activate asdun-cloud
uvicorn app:app --host 0.0.0.0 --port 8000
```

Or from the repository root. This command prefers mamba/conda and falls back to `.venv`:

```powershell
.\scripts\run_cloud_server.ps1
```

Useful variants:

```powershell
.\scripts\run_cloud_server.ps1 -SkipInstall
.\scripts\run_cloud_server.ps1 -UseVenv
.\scripts\run_cloud_server.ps1 -EnvName asdun-cloud
```

## Test

Health check from Windows:

```powershell
curl http://127.0.0.1:8000/health
```

Health check from Raspberry Pi:

```bash
curl http://asdun-cloud:8000/health
```

Recommended networking for the current project:

- Keep the Raspberry Pi hostname / Tailscale device name as `asdun`.
- Rename the Windows/NVIDIA Tailscale device to `asdun-cloud`.
- Keep the Raspberry Pi client config pointed at `http://asdun-cloud:8000`.
- Keep the cloud server binding to `0.0.0.0`; do not bind it to a temporary phone-hotspot IP.

From Windows, this helper verifies both local service health and the Pi-to-Windows route:

```powershell
.\scripts\check_pi_cloud_link.ps1 -PiHost asdun -PiUser pi -CloudUrl http://asdun-cloud:8000
```

When developing on Windows and testing on the Raspberry Pi, sync only the Pi-side source/config/model files:

```powershell
.\scripts\sync_to_pi.ps1 -PiHost asdun -PiUser pi -RemoteDir ~/asdun_pi
```

If Tailscale DNS is unstable, add the Windows Tailscale `100.x.y.z` address as a second entry under
`cloud_server_urls` in `config/app.yaml`.

Expected output includes:

```json
{
  "ok": true,
  "device": "cuda",
  "active_provider": "CUDAExecutionProvider"
}
```

If `device` is `cpu`, the service can still run Stage A, but GPU acceleration is not ready yet.

For Stage B, also check:

```json
{
  "emotion": {
    "ready": true,
    "provider": "CUDAExecutionProvider",
    "session_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]
  }
}
```

If `emotion.ready` is `false`, `/analyze` will still respond but will fall back to the previous fake emotion result and include `debug.emotion_error`.
If `emotion.provider` stays `CPUExecutionProvider` while `CUDAExecutionProvider` is available, inspect
`emotion.requested_providers`, `emotion.session_providers`, and `emotion.warning` in `/health`.
The server calls `onnxruntime.preload_dlls()` during startup, so CUDA/cuDNN runtime packages installed in the Python
environment can be discovered without editing the global Windows PATH.

## Tuning Notes

Cloud identity matching scores each person by the mean of the nearest `identity.score_top_k` samples instead of one
single nearest sample. Enrollment now also rejects obvious outliers before writing the cloud gallery. At recognition
time each person is represented by a clean sample set, a centroid template, and several pose-diverse templates; the
final score combines nearest-sample, centroid, and template distances. This is more conservative and helps avoid two
people being confused by one accidental outlier.
If many known faces become `Unknown`, inspect `id_dist` and `id_gap` in the Raspberry Pi logs before relaxing
`identity.match_threshold`, `identity.margin_threshold`, or `cloud_identity_min_confidence`.

If `id_gap=1.000`, it usually means the cloud gallery only has one person candidate, not that identity separation is
excellent. Check the gallery first:

```powershell
curl http://127.0.0.1:8000/gallery
```

For deeper gallery quality checks:

```powershell
curl http://127.0.0.1:8000/gallery/diagnostics
```

Look for non-empty `outliers` inside each person and `warning: "too_close"` in pair rows. Those indicate either a bad
enrollment sample or two identities that are too close in the current gallery.

The InsightFace backend uses a separate gallery file, `data/cloud_gallery_insightface.npz`. After switching identity
backends, enroll every person again from the Raspberry Pi menu so old embeddings are not mixed with the new model.

The FER+ emotion model often lets Calm dominate weak Sad expressions. The `sad_*` parameters add a dedicated low
intensity Sad path while keeping the normal Happy/Angry filters stricter.

## Gallery

List cloud gallery:

```powershell
curl http://127.0.0.1:8000/gallery
```

Enroll one person with several face crops:

```powershell
curl -X POST http://127.0.0.1:8000/gallery/enroll `
  -F "name=yuqin" `
  -F "replace=true" `
  -F "images=@C:\path\to\face1.jpg" `
  -F "images=@C:\path\to\face2.jpg"
```

Or use the helper script:

```powershell
python .\cloud_server\tools\enroll_gallery.py --name yuqin C:\path\to\yuqin_faces
```

In the full hybrid flow, Raspberry Pi enrollment also uploads the captured local enrollment images to this endpoint
automatically after local enrollment finishes.

Delete one person from the cloud gallery:

```powershell
curl -X POST http://127.0.0.1:8000/gallery/delete -F "name=yuqin"
```

The Raspberry Pi delete menu can delete local-only, cloud-only, or local+cloud data. Cloud-only deletion lists the
current cloud gallery first and falls back to manual name input only when the gallery cannot be fetched.

The cloud gallery is stored at `data/cloud_gallery.npz` and is intentionally ignored by git.
