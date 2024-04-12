import os
import boto3
from botocore.exceptions import NoCredentialsError

def upload_to_s3(local_folder_path, bucket_name, s3_folder_name):
    # Set your AWS credentials (replace 'your_access_key' and 'your_secret_key' with your actual credentials)
    aws_access_key = 'AKIA6GBMEWKXRC2LG7NJ'
    aws_secret_key = '9sxB6QAUiHGbiNeX2aXfF/0/vef3qyk8ptbd1OHx'

    # Create an S3 client
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)

    try:
        # Upload each file in the local folder to the specified S3 bucket and folder
        for root, dirs, files in os.walk(local_folder_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                s3_file_path = os.path.join(s3_folder_name, os.path.relpath(local_file_path, local_folder_path))

                s3.upload_file(local_file_path, bucket_name, s3_file_path)

        print("Upload successful.")
    except NoCredentialsError:
        print("Credentials not available or incorrect.")

# Example usage
local_folder_path = "C:/Users/shail/Dropbox/My PC (LAPTOP-674MEPPR)/Desktop/Final Year/Code/webscrape"
bucket_name = "omgadhave"
s3_folder_name = "folder"

upload_to_s3(local_folder_path, bucket_name, s3_folder_name)

