import zipfile
import os

print("Starting extraction...")
with zipfile.ZipFile('data.zip', 'r') as zip_ref:
    zip_ref.extractall('.')
print("Extraction complete!") 