import sys
sys.path.append('.')
from find_synsets_in_captions import find_synsets
import json
from get_dataset import get_orig_dataset
from tqdm import tqdm

dataset = sys.argv[1]

data = get_orig_dataset(dataset)
for i in tqdm(range(len(data)), desc=dataset):
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
