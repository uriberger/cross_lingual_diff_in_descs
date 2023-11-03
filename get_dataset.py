import json
import os

datasets = ['COCO', 'flickr30k', 'pascal_sentences', 'xm3600_ar', 'xm3600_bn', 'xm3600_cs', 'xm3600_da', 'xm3600_de',
            'xm3600_el', 'xm3600_es', 'xm3600_fa', 'xm3600_fi', 'xm3600_fil', 'xm3600_fr', 'xm3600_he', 'xm3600_hi',
            'xm3600_hr', 'xm3600_hu', 'xm3600_id', 'xm3600_it', 'xm3600_ja', 'xm3600_ko', 'xm3600_mi', 'xm3600_nl',
            'xm3600_no', 'xm3600_pl', 'xm3600_pt', 'xm3600_quz', 'xm3600_ro', 'xm3600_ru', 'xm3600_sv', 'xm3600_sw',
            'xm3600_te', 'xm3600_th', 'xm3600_tr', 'xm3600_uk', 'xm3600_vi', 'xm3600_zh', 'multi30k', 'STAIR-captions',
            'YJCaptions', 'coco-cn', 'flickr8kcn', 'ai_challenger']

def get_dataset(dataset_name):
    if dataset_name == 'COCO':
        with open('../CLIP_prefix_caption/dataset_coco.json', 'r') as fp:
            coco_data = json.load(fp)['images']
        data = []
        for x in coco_data:
            for y in x['sentences']:
                data.append({
                    'image_id': x['cocoid'],
                    'image_path': f'/cs/labs/oabend/uriber/datasets/COCO/{x["filepath"]}/{x["filename"]}',
                    'caption': y['raw']
                    })
    elif dataset_name == 'flickr30k':
        with open('/cs/labs/oabend/uriber/datasets/flickr30/karpathy/dataset_flickr30k.json', 'r') as fp:
            flickr_data = json.load(fp)['images']
        data = []
        for x in flickr_data:
            for y in x['sentences']:
                data.append({
                    'image_id': int(x['filename'].split('.jpg')[0]),
                    'image_path': f'/cs/labs/oabend/uriber/datasets/flickr30/images/{x["filename"]}',
                    'caption': y['raw']
                    })
    elif dataset_name == 'pascal_sentences':
        sdir = '/cs/labs/oabend/uriber/datasets/pascal_sentences/sentence'
        subdir_names = sorted(os.listdir(sdir))
        data = []
        for subdir_name in subdir_names:
            subdir_path = os.path.join(sdir, subdir_name)
            if not os.path.isdir(subdir_path):
                continue
            file_names = sorted(os.listdir(subdir_path))
            for file_name in file_names:
                file_path = os.path.join(subdir_path, file_name)
                with open(file_path, 'r') as fp:
                    for line in fp:
                        caption = line.strip()
                        data.append({'image_path': f'/cs/labs/oabend/uriber/datasets/pascal_sentences/dataset/{subdir_name}/{file_name.replace(".txt", ".jpg")}', 'caption': caption})
    elif dataset_name.startswith('xm3600'):
        lang = dataset_name.split('xm3600_')[1]
        if lang == 'en':
            with open('xm3600/xm3600_en.json', 'r') as fp:
                data = json.load(fp)
        else:
            with open(f'xm3600/xm3600_{lang}_to_en.json', 'r') as fp:
                data = json.load(fp)
            with open(f'xm3600/xm3600_{lang}.json', 'r') as fp:
                orig_data = json.load(fp)
            assert len(data) == len(orig_data)
            for i in range(len(data)):
                data[i]['orig'] = orig_data[i]['caption']
        for i in range(len(data)):
            data[i]['image_path'] = f'/cs/labs/oabend/uriber/datasets/crossmodal3600/images/{hex(data[i]["image_id"])[2:].zfill(16)}.jpg'
    elif dataset_name == 'multi30k':
        with open('multi30k_en.json', 'r') as fp:
            data = json.load(fp)
        with open('../playground/multi30k_caption.json', 'r') as fp:
            orig_data = json.load(fp)
        for i in range(len(data)):
            data[i]['orig'] = orig_data[i]['caption']
            data[i]['image_path'] = f'/cs/labs/oabend/uriber/datasets/flickr30/images/{data[i]["image_id"]}.jpg'
    elif dataset_name == 'STAIR-captions':
        with open('../CLIP_prefix_caption/dataset_coco.json', 'r') as fp:
            coco_data = json.load(fp)['images']
        iid_to_split = {x['cocoid']: 'train' if x['split'] == 'train' else 'val' for x in coco_data}
        with open('../playground/STAIR_captions.json', 'r') as fp:
            orig_data = json.load(fp)
        with open('../playground/STAIR_translated.json', 'r') as fp:
            tran_data = json.load(fp)
        data = []
        for i in range(len(orig_data)):
            image_id = orig_data[i]['image_id']
            split = iid_to_split[image_id]
            data.append({
                'image_id': image_id,
                'caption': tran_data[i]['translatedText'],
                'orig': orig_data[i]['caption'],
                'image_path': f'/cs/labs/oabend/uriber/datasets/COCO/{split}2014/COCO_{split}2014_{str(image_id).zfill(12)}.jpg'
            })
    elif dataset_name == 'YJCaptions':
        with open('../CLIP_prefix_caption/dataset_coco.json', 'r') as fp:
            coco_data = json.load(fp)['images']
        iid_to_split = {x['cocoid']: 'train' if x['split'] == 'train' else 'val' for x in coco_data}
        with open('../playground/YJCaptions.json', 'r') as fp:
            orig_data = json.load(fp)
        with open('../playground/YJ_translated.json', 'r') as fp:
            tran_data = json.load(fp)
        data = []
        for i in range(len(orig_data)):
            image_id = orig_data[i]['image_id']
            split = iid_to_split[image_id]
            data.append({
                'image_id': image_id,
                'caption': tran_data[i]['translatedText'],
                'orig': orig_data[i]['caption'],
                'image_path': f'/cs/labs/oabend/uriber/datasets/COCO/{split}2014/COCO_{split}2014_{str(image_id).zfill(12)}.jpg'
            })
    elif dataset_name == 'coco-cn':
        with open('../CLIP_prefix_caption/dataset_coco.json', 'r') as fp:
            coco_data = json.load(fp)['images']
        iid_to_split = {x['cocoid']: 'train' if x['split'] == 'train' else 'val' for x in coco_data}
        with open('../playground/coco_cn_captions.json', 'r') as fp:
            orig_data = json.load(fp)
        with open('../playground/coco_cn_translated.json', 'r') as fp:
            tran_data = json.load(fp)
        data = []
        for i in range(len(orig_data)):
            image_id = orig_data[i]['image_id']
            split = iid_to_split[image_id]
            data.append({
                'image_id': image_id,
                'caption': tran_data[i]['translatedText'],
                'orig': orig_data[i]['caption'],
                'image_path': f'/cs/labs/oabend/uriber/datasets/COCO/{split}2014/COCO_{split}2014_{str(image_id).zfill(12)}.jpg'
            })
    elif dataset_name == 'flickr8kcn':
        with open('../playground/flickr8kcn_captions.json', 'r') as fp:
            orig_data = json.load(fp)
        with open('../playground/flickr8kcn_translated.json', 'r') as fp:
            tran_data = json.load(fp)
        data = []
        for i in range(len(orig_data)):
            image_id = orig_data[i]['image_id']
            data.append({
                'image_id': image_id,
                'caption': tran_data[i]['translatedText'],
                'orig': orig_data[i]['caption'],
                'image_path': f'/cs/labs/oabend/uriber/datasets/flickr30/images/{image_id}.jpg'
            })
    elif dataset_name == 'ai_challenger':
        with open('/cs/labs/oabend/uriber/datasets/ai_challenger/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json', 'r') as fp:
            aic_data = json.load(fp)
        with open('../playground/aic_captions.json', 'r') as fp:
            orig_data = json.load(fp)
        with open('../playground/aic_translated.json', 'r') as fp:
            tran_data = json.load(fp)
        data = []
        to_reduce = 0
        image_dir = 'ai_challenger_caption_train_20170902/caption_train_images_20170902'
        for i in range(len(tran_data)):
            if i == len(aic_data) * 5:
                with open('/cs/labs/oabend/uriber/datasets/ai_challenger/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json', 'r') as fp:
                    aic_data = json.load(fp)
                to_reduce = i
                image_dir = 'ai_challenger_caption_validation_20170910/caption_validation_images_20170910'
            sample_ind = (i - to_reduce) // 5
            image_id = int(aic_data[sample_ind]['image_id'].split('.jpg')[0], 16)
            data.append({
                'image_id': image_id,
                'caption': tran_data[i]['translatedText'],
                'orig': orig_data[i]['caption'],
                'image_path': f'/cs/labs/oabend/uriber/datasets/ai_challenger/{image_dir}/{aic_data[sample_ind]["image_id"]}'
            })
    else:
        assert False, f'Unknown dataset {dataset_name}'

    for i in range(len(data)):
        data[i]['source'] = dataset_name

    return data
