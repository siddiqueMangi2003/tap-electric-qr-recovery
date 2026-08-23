from abc import ABC, abstractmethod


class ObjectStorage(ABC):
    """Storage boundary that keeps image blobs out of PostgreSQL."""

    @abstractmethod
    def upload_image(self, scan_id: str, data: bytes, content_type: str) -> str:
        """Persist image bytes and return an opaque URI."""

    @abstractmethod
    def get_image(self, image_uri: str) -> bytes:
        """Read image bytes by URI."""

    @abstractmethod
    def delete_image(self, image_uri: str) -> None:
        """Delete image bytes, used for retention and failed-write compensation."""
