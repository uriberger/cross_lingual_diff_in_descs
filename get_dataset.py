import json
from config import coco_json_path, stair_json_path, stair_translated_json_path

datasets = ['COCO', 'xm3600_ar', 'xm3600_bn', 'xm3600_cs', 'xm3600_da', 'xm3600_de', 'xm3600_el', 'xm3600_en',
            'xm3600_es', 'xm3600_fa', 'xm3600_fi', 'xm3600_fil', 'xm3600_fr', 'xm3600_he', 'xm3600_hi', 'xm3600_hr',
            'xm3600_hu', 'xm3600_id', 'xm3600_it', 'xm3600_ja', 'xm3600_ko', 'xm3600_mi', 'xm3600_nl', 'xm3600_no',
            'xm3600_pl', 'xm3600_pt', 'xm3600_quz', 'xm3600_ro', 'xm3600_ru', 'xm3600_sv', 'xm3600_sw', 'xm3600_te',
            'xm3600_th', 'xm3600_tr', 'xm3600_uk', 'xm3600_vi', 'xm3600_zh', 'STAIR-captions']

def get_processed_dataset(dataset_name):
    with open(f'datasets/{dataset_name}.json', 'r') as fp:
        data = json.load(fp)

    return data

def get_orig_dataset(dataset_name):
    if dataset_name == 'COCO':
        with open(coco_json_path, 'r') as fp:
            coco_data = json.load(fp)['images']
        data = []
        for x in coco_data:
            for y in x['sentences']:
                data.append({
                    'image_id': x['cocoid'],
                    'caption': y['raw']
                    })
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
    elif dataset_name == 'STAIR-captions':
        with open(coco_json_path, 'r') as fp:
            coco_data = json.load(fp)['images']
        iid_to_split = {x['cocoid']: 'train' if x['split'] == 'train' else 'val' for x in coco_data}
        with open(stair_json_path, 'r') as fp:
            orig_data = json.load(fp)
        with open(stair_translated_json_path, 'r') as fp:
            tran_data = json.load(fp)
        data = []
        for i in range(len(orig_data)):
            image_id = orig_data[i]['image_id']
            split = iid_to_split[image_id]
            data.append({
                'image_id': image_id,
                'caption': tran_data[i]['translatedText'],
                'orig': orig_data[i]['caption'],
            })
    else:
        assert False, f'Unknown dataset {dataset_name}'

    for i in range(len(data)):
        data[i]['source'] = dataset_name

    return data
