from app.services.storage.base import ObjectStorage


class S3ObjectStorage(ObjectStorage):
    """S3-compatible adapter; boto3 is imported only when this backend is selected."""

    def __init__(self, bucket: str, endpoint_url: str | None = None) -> None:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - optional integration
            raise RuntimeError("Install the project with the 's3' extra to use S3") from exc
        self.bucket = bucket
        self.client = boto3.client("s3", endpoint_url=endpoint_url)

    def upload_image(self, scan_id: str, data: bytes, content_type: str) -> str:
        extension = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}.get(
            content_type, "bin"
        )
        key = f"scans/{scan_id}/original.{extension}"
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)
        return f"s3://{self.bucket}/{key}"

    def get_image(self, image_uri: str) -> bytes:
        key = self._key(image_uri)
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    def delete_image(self, image_uri: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self._key(image_uri))

    def _key(self, image_uri: str) -> str:
        prefix = f"s3://{self.bucket}/"
        if not image_uri.startswith(prefix):
            raise ValueError("Object URI does not belong to the configured bucket")
        return image_uri.removeprefix(prefix)
