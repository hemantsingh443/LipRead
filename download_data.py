import gdown

url = 'https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL'
output = 'data.zip'

print("Downloading data.zip...")
gdown.download(url, output, quiet=False)

print("Extracting data.zip...")
gdown.extractall('data.zip')

print("Download and extraction complete!") 