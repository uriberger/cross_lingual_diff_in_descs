import torch
import clip
from PIL import Image, ImageDraw
from dataset_builders.coco_dataset_builders.coco_dataset_builder import CocoDatasetBuilder
import random

dataset_builder = CocoDatasetBuilder('/cs/labs/oabend/uriber/datasets/COCO', 'None', 1)
print('Generating caption data...')
caption_data = dataset_builder.get_caption_data()
print('Generating bboxes data...')
_, gt_bboxes_data = dataset_builder.get_gt_classes_bboxes_data()
image_path_finder = dataset_builder.get_image_path_finder()

def hide_obj(image_id, bbox):
    image_path = image_path_finder.get_image_path(image_id)
    image_obj = Image.open(image_path)
    draw_obj = ImageDraw.Draw(image_obj)
    draw_obj.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], fill='black', outline='black')

    return image_obj

image_ids = list(set([x['image_id'] for x in caption_data]))
images_with_multiple_objects = [x for x in image_ids if x in gt_bboxes_data and len(gt_bboxes_data[x]) > 1]
selected_image_ids = random.sample(images_with_multiple_objects, 20)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_id_to_logits = {}
count = 0

for image_id in selected_image_ids:
    count += 1
    print('Starting image ' + str(count) + ' out of ' + str(len(selected_image_ids)), flush=True)
    image_id_to_logits[image_id] = {}
    orig_image = preprocess(Image.open(image_path_finder.get_image_path(image_id))).unsqueeze(0).to(device)
    captions = [x['caption'] for x in caption_data if x['image_id'] == image_id]
    text = clip.tokenize(captions).to(device)

    with torch.no_grad():
        orig_logits, _ = model(orig_image, text)

    image_id_to_logits[image_id]['orig'] = orig_logits
    image_id_to_logits[image_id]['adjusted'] = []

    for bbox in gt_bboxes_data[image_id]:
        cur_image_obj = hide_obj(image_id, bbox)
        adjusted_image = preprocess(cur_image_obj).unsqueeze(0).to(device)

        with torch.no_grad():
            adjusted_logits, _ = model(adjusted_image, text)

        image_id_to_logits[image_id]['adjusted'].append(adjusted_logits)

torch.save(image_id_to_logits, 'image_id_to_logits')
    
