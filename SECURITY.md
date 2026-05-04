# Security Notes

Do not commit runtime secrets, personal data, or model/data artifacts.

`config/app.yaml` is intentionally treated as a private mixed runtime file. It
contains both shareable tuning values and private deployment identity.

Safe to publish in `config/app.example.yaml` if desired:

- Numeric thresholds, cooldowns, crop scales, frame sizes, and scheduler values.
- Generic model path names, as long as the model weight files are not committed.
- Generic device IDs or placeholders used only as examples.

Keep private in `config/app.yaml`:

- Device tokens, admin tokens, passwords, API keys, and `.env` values.
- Public server IPs/domains before you are ready to share them.
- Tailscale/private IPs, personal hostnames, local usernames, and real display names.
- Face gallery files, databases, telemetry history, screenshots with personal data,
  and any model weights or tuned artifacts you do not want public.

Keep these files local:

- `config/app.yaml`
- `cloud_server/config.yaml`
- `deploy/platform_server/platform.env`
- `data/`
- `platform_server/data/`
- `cloud_server/data/`
- `backups/`
- model weights under `models/`

Use these templates instead:

- `config/app.example.yaml`
- `cloud_server/config.example.yaml`
- `deploy/platform_server/platform.env.example`

Rotate tokens immediately if they are pasted into chat, committed, pushed, or shown in a public screenshot.

Recommended token sources:

```bash
openssl rand -hex 16
openssl rand -hex 32
```

Before publishing:

```bash
git status --short
git ls-files -c -i --exclude-standard
```

The second command should print nothing.
