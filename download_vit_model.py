import zipfile
import os

# download the zipped folder that contains onnx files from google drive
fileid = "17pqu4WE-aYEwdNgVILGSWKdlDAf4NFT-"
filename = "aebs_vit.zip"
os.system(f"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={fileid}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={fileid}" -O {filename} && rm -rf /tmp/cookies.txt""")

# upzip the folder
with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall('./models/')
# remove the zipped folder
os.system(f"""rm {filename}""")