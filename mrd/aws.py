import os

import boto3


def s3_client():
    """S3 client for the DEP buckets, which are all in us-west-2.

    Uses the AWS_PROFILE profile when it's set, even if AWS_ACCESS_KEY_ID etc. are also set (boto3 would otherwise
    prefer those). Unset, e.g. in Argo pods, the default credential chain applies.
    """
    return boto3.Session(profile_name=os.environ.get("AWS_PROFILE")).client("s3", region_name="us-west-2")
