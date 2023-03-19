import torch
import clip
import os
from PIL import Image, ImageDraw
import json

image_dir = '/cs/labs/oabend/uriber/datasets/COCO/train2014'

def get_image_path(image_id):
    return os.path.join(image_dir, 'COCO_train2014_' + str(image_id).zfill(12) + '.jpg')

def hide_obj(image_id, bbox):
    image_path = get_image_path(image_id)
    image_obj = Image.open(image_path)
    draw_obj = ImageDraw.Draw(image_obj)
    draw_obj.rectangle([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], fill='black', outline='black')

    return image_obj

image_obj1 = hide_obj(480023, [116.95, 305.86, 285.3, 266.03])
image_obj2 = hide_obj(480023, [75.23, 134.7, 203.17, 215.63])
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

orig_image = preprocess(Image.open(get_image_path(480023))).unsqueeze(0).to(device)
image1 = preprocess(image_obj1).unsqueeze(0).to(device)
image2 = preprocess(image_obj2).unsqueeze(0).to(device)
captions = ['A hand holding a hot dog in a paper container covered in mustard and ketchup.', 'A person holding a hot dog with yellow mustard and onions on it, at a sports stadium.', 'A hand holds a traditional loaded ballgame hotdog.', 'A person holding up a hot dog at a ball park.', 'A close-up of a person holding hot dog to the camera.']
text = clip.tokenize(captions).to(device)

with torch.no_grad():
    orig_logits, _ = model(orig_image, text)
    logits1, _ = model(image1, text)
    logits2, _ = model(image2, text)

print(orig_logits)
print(logits1)
print(logits2)
    