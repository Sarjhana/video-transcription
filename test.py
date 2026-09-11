import os
from pyannote.audio import Model

HF_TOKEN = os.environ["HF_TOKEN"]

model = Model.from_pretrained("pyannote/segmentation", use_auth_token=HF_TOKEN)
print("Segmentation model loaded successfully.")