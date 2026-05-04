# ASDUN Raspberry Pi Face Recognition and Emotion Detection

ASDUN is an edge-plus-cloud face recognition and emotion detection project built around a Raspberry Pi camera, a Windows/NVIDIA GPU inference service, a FastAPI platform server, and optional ESP32 telemetry devices.

The project currently focuses on a practical prototype:

- Raspberry Pi runs the camera, local face pipeline, tracking, enrollment, and UI loop.
- Windows/NVIDIA host can run cloud inference for heavier identity/emotion recognition.
- Platform Server receives device status, recognition events, ESP32 telemetry, and device commands.
- Web dashboard shows device state, people statistics, emotion share, telemetry, commands, and mood reports.
- ESP32 can upload temperature, humidity, light, RSSI, and receive platform commands.

## Architecture

```text
Raspberry Pi camera
  -> C++ app / NCNN local inference
  -> optional Windows GPU cloud inference
  -> Platform Server status + recognition event upload

Windows GPU cloud server
  -> FastAPI / ONNX Runtime / InsightFace
  -> identity + emotion analysis
  -> optional Platform Server event upload

Platform Server
  -> FastAPI + SQLite + static dashboard
  -> device status, people summary, commands, telemetry, mood report

ESP32
  -> telemetry upload
  -> command polling/result reporting
```

## Repository Layout

```text
src/                  C++ Raspberry Pi application implementation
include/              C++ headers
config/               Runtime config template
cloud_server/         Windows/NVIDIA cloud inference FastAPI service
platform_server/      Platform FastAPI service and dashboard
deploy/platform_server/
                      Docker, systemd, Nginx, and cloud deployment files
examples/             ESP32 example client
models/               Model placement notes, weights are not committed
scripts/              Helper scripts for run, sync, test, and deployment
data/                 Local runtime data, ignored by Git
network.md            Network stage notes
up.md                 Platform upgrade plan
use.md                End-to-end usage notes
esp32.md              ESP32 integration notes
SECURITY.md           Privacy and publishing rules
```

## Privacy Boundary

Real runtime files are intentionally ignored by Git:

- `config/app.yaml`
- `cloud_server/config.yaml`
- `deploy/platform_server/platform.env`
- `data/`
- `platform_server/data/`
- `cloud_server/data/`
- `backups/`
- model weights under `models/`

Use the safe templates instead:

```powershell
Copy-Item .\config\app.example.yaml .\config\app.yaml
Copy-Item .\cloud_server\config.example.yaml .\cloud_server\config.yaml
Copy-Item .\deploy\platform_server\platform.env.example .\deploy\platform_server\platform.env
```

Numeric thresholds, crop scales, cooldowns, frame sizes, and scheduling values in example configs can be public if you want others to reproduce the behavior. Tokens, public server addresses, Tailscale IPs, real hostnames, usernames, face galleries, databases, logs, and private model weights should stay local.

Before publishing, read [SECURITY.md](SECURITY.md).

## Raspberry Pi App

Install the native dependencies on Raspberry Pi first, including OpenCV, SQLite, GStreamer, libcurl, and NCNN. Then configure and build:

```bash
cmake -S . -B build_rpi -DCMAKE_BUILD_TYPE=Release
cmake --build build_rpi -j4
```

Run the tuned helper:

```bash
./scripts/run_rpi_turbo.sh
```

Typical app menu:

```text
1) Enroll or update person
2) Live recognition + emotion
3) Delete person
0) Exit
```

Edit `config/app.yaml` for camera, thresholds, cloud inference, and Platform Server settings.

## Cloud Inference Server

The cloud server is usually run on a Windows/NVIDIA machine:

```powershell
conda env create -f .\cloud_server\environment.yml
conda activate asdun-cloud
.\scripts\run_cloud_server.ps1
```

Health check:

```powershell
curl.exe http://127.0.0.1:8000/health
```

More details are in [cloud_server/README.md](cloud_server/README.md).

## Platform Server

Run locally for development:

```powershell
.\scripts\run_platform_server.ps1
```

Open:

```text
http://127.0.0.1:9000
```

Health check:

```powershell
curl.exe http://127.0.0.1:9000/health
```

Run on a cloud server with Docker:

```bash
cd /opt/asdun_pi/deploy/platform_server
cp platform.env.example platform.env
nano platform.env
docker-compose up -d --build
```

More deployment details are in [deploy/platform_server/README.md](deploy/platform_server/README.md).

## ESP32 Integration

The ESP32 side should post telemetry to:

```text
POST /api/telemetry
```

It can also poll and complete commands:

```text
GET /api/commands/pending
POST /api/commands/{command_id}/result
```

See [esp32.md](esp32.md) and [examples/esp32_platform_client](examples/esp32_platform_client).

## Common Platform Tests

Post a Pi status sample:

```powershell
.\scripts\post_platform_status.ps1 -Preset pi
```

Post telemetry sample:

```powershell
.\scripts\test_platform_telemetry.ps1 -PlatformUrl http://127.0.0.1:9000 -DeviceId esp32-01
```

Post a command and wait for result:

```powershell
.\scripts\post_platform_command.ps1 -PlatformUrl http://127.0.0.1:9000 -DeviceId pi-01 -Command ping
```

If admin auth is enabled, add:

```powershell
-AdminToken your-admin-token
```

## Publishing Notes

This repository has runtime data and model files that should not be public. A normal commit after the privacy cleanup removes them from the current Git tree, but old Git history may still contain them if they were committed before.

For a public GitHub release, the safest approach is to create a clean public repository from the current sanitized tree instead of pushing old history.

First commit the sanitized state locally:

```powershell
git status --short
git ls-files -c -i --exclude-standard
git add .
git commit -m "Prepare public release"
```

The second command should print nothing. If it prints ignored runtime files that are still tracked, fix that before publishing.

Then create a clean public repository without old history:

```powershell
New-Item -ItemType Directory ..\asdun_pi_public
git archive --format=tar HEAD | tar -x -C ..\asdun_pi_public
cd ..\asdun_pi_public
git init
git branch -M main
git add .
git commit -m "Initial public release"
git remote add origin https://github.com/<your-name>/<your-public-repo>.git
git push -u origin main
```

If your current Git history is already clean and private files were never committed, you can push the normal branch instead:

```powershell
git push -u origin feature/web-build
```

For this project, prefer the clean public repository path unless you have already cleaned Git history.

## Documentation

- [use.md](use.md): practical usage flow and troubleshooting
- [network.md](network.md): network/platform task stages
- [up.md](up.md): platform upgrade roadmap
- [performance.md](performance.md): Raspberry Pi performance notes
- [cloud_server/README.md](cloud_server/README.md): cloud inference service
- [platform_server/README.md](platform_server/README.md): platform API and dashboard
- [deploy/platform_server/README.md](deploy/platform_server/README.md): cloud deployment
- [SECURITY.md](SECURITY.md): privacy and secret handling
