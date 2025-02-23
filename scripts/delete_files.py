from openai import AzureOpenAI
import os 
import json
import time
import datetime

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-10-21",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def main():
    response = client.files.list()
    files = list(response.data)
    if len(files) == 0:
        print("No files found.")
        return
    for i, file in enumerate(files, start=1):
        created_date = datetime.datetime.utcfromtimestamp(file.created_at).strftime('%Y-%m-%d')
        print(f"[{i}] {file.filename} [{file.id}], Created: {created_date}")

    for i in range(10,368):
        selected_file = files[i]
        client.files.delete(selected_file.id)
        print(f"File deleted: {selected_file.filename}")

main()