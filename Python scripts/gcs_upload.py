
from google.cloud import storage

# Initialize a client
client = storage.Client.from_service_account_json('C:/Users/ASUS/Desktop/orange_telecom_project/service_account.json')


# Specify the bucket name
bucket_name = 'your-bucket-name'
bucket = client.bucket(bucket_name)

# Specify the file to upload
file_path = 'C:/Users/ASUS/Desktop/orange_telecom_project/service_account.json'
blob_name = 'service_account.json'

# Upload the file
blob = bucket.blob(blob_name)
blob.upload_from_filename(file_path)

print(f"File {file_path} uploaded to {bucket_name}/{blob_name}.")
