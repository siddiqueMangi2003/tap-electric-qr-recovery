from pathlib import Path

from app.services.storage.base import ObjectStorage

_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


class LocalObjectStorage(ObjectStorage):
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def upload_image(self, scan_id: str, data: bytes, content_type: str) -> str:
        suffix = _EXTENSIONS.get(content_type, ".bin")
        path = self.root / f"{scan_id}{suffix}"
        path.write_bytes(data)
        return f"local://{path.name}"

    def get_image(self, image_uri: str) -> bytes:
        return self._resolve_uri(image_uri).read_bytes()

    def delete_image(self, image_uri: str) -> None:
        path = self._resolve_uri(image_uri)
        if path.exists():
            path.unlink()

    def _resolve_uri(self, image_uri: str) -> Path:
        if not image_uri.startswith("local://"):
            raise ValueError("Unsupported local object URI")
        name = image_uri.removeprefix("local://")
        if Path(name).name != name:
            raise ValueError("Invalid object key")
        path = (self.root / name).resolve()
        if path.parent != self.root:
            raise ValueError("Object URI escapes storage root")
        return path
