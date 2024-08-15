from collections import defaultdict
import csv
import json
from get_dataset import datasets
from config import use_low_resource_langs

langs = [x.split('_')[1] for x in datasets if x.startswith('xm3600_')]
low_resource_langs = ['bn', 'te', 'sw', 'quz', 'mi']
if not use_low_resource_langs:
    langs = [x for x in langs if x not in low_resource_langs]

east_asian_langs = ['zh', 'ja', 'ko', 'th', 'vi', 'fil', 'id']

# Root phrases:
#   person, vehicle, furniture, animal, food, bag, clothing, tableware, plant, electronic_equipment, home_appliance,
#   toy, building, mountain, kitchen_utensil, sky, sun, body_part, body_of_water, hand_tool, musical_instrument,
#   writing_implement, jewelry, weapon, timepiece
with open('phrase2synsets.json', 'r') as fp:
    phrase2synsets = json.load(fp)

with open('phrase2hypernym.json', 'r') as fp:
    phrase2hypernym = json.load(fp)

with open('synsets_c2p.json', 'r') as fp:
    child2parent = json.load(fp)

parent2children = defaultdict(list)
for child, parent in child2parent.items():
    parent2children[parent].append(child)

with open('implicit_synsets.json', 'r') as fp:
    implicit_synsets = json.load(fp)

all_synsets = set([x for outer in phrase2synsets.values() for x in outer if x is not None]).union(implicit_synsets).union(child2parent)

def is_hyponym_of(synset1, synset2):
    if synset1 == synset2:
        return True
    if synset1 in child2parent:
        return is_hyponym_of(child2parent[synset1], synset2)
    return False

def get_image_id_to_root_synsets():
    csv_path = 'xm3600_annotation.csv'
    iid2root_synset = {}
    with open(csv_path, 'r') as fp:
        my_reader = csv.reader(fp)
        res = list(my_reader)

    for sample in res[1:]:
        file_name = sample[0]
        iid = int(file_name, 16)
        assert iid not in iid2root_synset
        iid2root_synset[iid] = []
        for ind in range(1, len(sample)):
            if sample[ind] == '1':
                iid2root_synset[iid].append(res[0][ind])

    return iid2root_synset

def verify_synset_in_image(synset, image_id, iid2root_synset):
    return image_id not in iid2root_synset or len([root_synset for root_synset in iid2root_synset[image_id] if is_hyponym_of(synset, root_synset)]) > 0

def get_synset_to_image_prob(dataset, filter_by_iid2root_dataset=True):
    iid2root_synset = get_image_id_to_root_synsets()

    with open(f'datasets/{dataset}.json', 'r') as fp:
        data = json.load(fp)
    synset_to_image_count = {x: defaultdict(int) for x in all_synsets}
    image_count = defaultdict(int)
    for sample in data:
        if 'synsets' not in sample or sample['synsets'] is None:
            continue
        image_count[sample['image_id']] += 1
        identified_synsets = []
        for synset in list(set([x[3] for x in sample['synsets']])):
            identified_synsets.append(synset)
            inner_synset = synset
            while inner_synset in child2parent:
                inner_synset = child2parent[inner_synset]
                identified_synsets.append(inner_synset)
        identified_synsets = list(set(identified_synsets))
        if filter_by_iid2root_dataset and sample['image_id'] in iid2root_synset:
            identified_synsets = [synset for synset in identified_synsets if verify_synset_in_image(synset, sample['image_id'], iid2root_synset)]
        for id_synset in identified_synsets:
            synset_to_image_count[id_synset][sample['image_id']] += 1
    synset_to_image_prob = {x[0]: {y[0]: y[1]/image_count[y[0]] for y in x[1].items()} for x in synset_to_image_count.items()}

    return synset_to_image_prob, synset_to_image_count, image_count
