import boto3


def s3_client():
    """S3 client for the DEP buckets, which are all in us-west-2."""
    return boto3.client("s3", region_name="us-west-2")
