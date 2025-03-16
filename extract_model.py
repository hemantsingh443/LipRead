import zipfile
import os
import shutil

# First, clean the models directory
if os.path.exists('models'):
    shutil.rmtree('models')
os.makedirs('models')

# Extract the checkpoint 96 zip file
print("Extracting model checkpoint...")
with zipfile.ZipFile('models - checkpoint 96.zip', 'r') as zip_ref:
    zip_ref.extractall('models')
print("Model extraction complete!") 