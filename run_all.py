from find_synsets_in_captions import find_synsets
import random
import json
from get_dataset import get_dataset, datasets
from tqdm import tqdm
import os
for dataset in [x for x in datasets if x.startswith('xm3600_')]:
#for dataset in ['COCO', 'STAIR-captions', 'YJCaptions', 'flickr8kcn', 'flickr30k', 'multi30k', 'coco-cn']:
    data = []
    if os.path.isfile(f'datasets/{dataset}.json'):
        with open(f'datasets/{dataset}.json', 'r') as fp:
            data = json.load(fp)
    samples_done_so_far = len(data)
    all_data = get_dataset(dataset)
    data = data + all_data[samples_done_so_far:]
    for i in tqdm(range(samples_done_so_far, len(data)), desc=dataset):
        if i % 10000 == 0:
            with open(f'datasets/{dataset}.json', 'w') as fp:
                fp.write(json.dumps(data[:i]))
        try:
            res = find_synsets(data[i]['caption'])
        except:
            assert False, f'Failed in dataset {dataset} in sample {i}'
        if res is None:
            if '\n' in data[i]['caption']:
                data[i]['synsets'] = None
            else:
                assert False, f'Got None in dataset {dataset} in sample {i}'
        else:
            data[i]['synsets'] = [x for x in res if x[3] is not None]
    with open(f'datasets/{dataset}.json', 'w') as fp:
        fp.write(json.dumps(data))
