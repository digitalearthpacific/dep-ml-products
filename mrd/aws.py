import boto3
from botocore.exceptions import ClientError


def s3_client():
    """S3 client for the DEP buckets, which are all in us-west-2."""
    return boto3.client("s3", region_name="us-west-2")


def s3_exists(client, bucket: str, key: str) -> bool:
    """Whether the key exists. Only "not found" counts as missing; any other error (e.g. 403) is raised.

    Strict on purpose, for checks where a false "missing" would overwrite something. Without s3:ListBucket, S3 answers
    403 for a missing key, so this raises then.
    """
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return False
        raise
