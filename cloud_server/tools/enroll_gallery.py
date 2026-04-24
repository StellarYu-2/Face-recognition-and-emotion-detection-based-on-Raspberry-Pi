from __future__ import annotations

import argparse
import json
import mimetypes
import uuid
from pathlib import Path
from urllib import request


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(paths: list[str]) -> list[Path]:
    images: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            images.extend(
                sorted(
                    child
                    for child in path.rglob("*")
                    if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
                )
            )
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    return images


def build_multipart(name: str, replace: bool, images: list[Path]) -> tuple[bytes, str]:
    boundary = f"----asdun-{uuid.uuid4().hex}"
    body = bytearray()

    def add_field(field_name: str, value: str) -> None:
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{field_name}"\r\n\r\n'.encode())
        body.extend(value.encode("utf-8"))
        body.extend(b"\r\n")

    add_field("name", name)
    add_field("replace", "true" if replace else "false")

    for image in images:
        mime = mimetypes.guess_type(image.name)[0] or "application/octet-stream"
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            (
                f'Content-Disposition: form-data; name="images"; filename="{image.name}"\r\n'
                f"Content-Type: {mime}\r\n\r\n"
            ).encode()
        )
        body.extend(image.read_bytes())
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode())
    return bytes(body), boundary


def main() -> int:
    parser = argparse.ArgumentParser(description="Enroll face images into the ASDUN cloud gallery.")
    parser.add_argument("--server", default="http://127.0.0.1:8000", help="Cloud server base URL.")
    parser.add_argument("--name", required=True, help="Person name.")
    parser.add_argument("--append", action="store_true", help="Append samples instead of replacing existing samples.")
    parser.add_argument("images", nargs="+", help="Image files or directories.")
    args = parser.parse_args()

    images = collect_images(args.images)
    if not images:
        raise SystemExit("No image files found.")

    body, boundary = build_multipart(args.name, replace=not args.append, images=images)
    url = args.server.rstrip("/") + "/gallery/enroll"
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with request.urlopen(req, timeout=120) as resp:
        payload = resp.read().decode("utf-8", errors="replace")
    print(json.dumps(json.loads(payload), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
