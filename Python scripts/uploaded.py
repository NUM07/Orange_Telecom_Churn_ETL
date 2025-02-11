from google.cloud import storage

# Initialize GCS client
client = storage.Client.from_service_account_json('service_account.json')
bucket_name = 'orange-telecom-data-lake'
bucket = client.bucket(bucket_name)

# Upload Kaggle dataset
kaggle_file = 'C:/Users/ASUS/Desktop/orange_telecom_project/service_account.json'
blob = bucket.blob('raw_data/churn-80.csv')
blob.upload_from_filename(kaggle_file)
print("Kaggle data uploaded to GCS")

# Upload Twitter data
twitter_file = '../Raw_Data/twitter_data.csv'
blob = bucket.blob('raw_data/twitter_data.csv')
blob.upload_from_filename(twitter_file)
print("Twitter data uploaded to GCS")
