from find_classes_in_caption import *
from collections import defaultdict
import json
import scipy.stats as stats
from scipy import spatial
import numpy as np
from get_dataset import datasets as all_datasets
import pandas as pd
from irrCAC.raw import CAC
from tqdm import tqdm
from sklearn.cluster import SpectralClustering

def get_class_to_image_prob(dataset):
    all_classes = list(set(word_classes + list(parent_to_children.keys())))
    with open(f'datasets/{dataset}.json', 'r') as fp:
        data = json.load(fp)
    class_to_image_count = {x: defaultdict(int) for x in all_classes}
    image_count = defaultdict(int)
    for sample in data:
        if 'classes' not in sample or sample['classes'] is None:
            continue
        image_count[sample['image_id']] += 1
        identified_classes = []
        for cur_class in list(set(sample['classes'])):
            identified_classes.append(cur_class)
            inner_cur_class = cur_class
            while inner_cur_class in child_to_parent:
                inner_cur_class = child_to_parent[inner_cur_class]
                identified_classes.append(inner_cur_class)
        identified_classes = list(set(identified_classes))
        for id_class in identified_classes:
            class_to_image_count[id_class][sample['image_id']] += 1
    class_to_image_prob = {x[0]: {y[0]: y[1]/image_count[y[0]] for y in x[1].items()} for x in class_to_image_count.items()}

    return class_to_image_prob, class_to_image_count, image_count

def get_class_to_image_prob_dataset_pair(datasets):
    all_classes = list(set(word_classes + list(parent_to_children.keys())))
    with open(f'datasets/{datasets[0]}.json', 'r') as fp:
        data1 = json.load(fp)
    with open(f'datasets/{datasets[1]}.json', 'r') as fp:
        data2 = json.load(fp)
    image_ids = set([x['image_id'] for x in data1]).intersection([x['image_id'] for x in data2])
    data1 = [x for x in data1 if x['image_id'] in image_ids and x['classes'] is not None]
    data2 = [x for x in data2 if x['image_id'] in image_ids and x['classes'] is not None]
    data = [data1, data2]
    class_to_image_count = []
    image_count = []
    class_to_image_prob = []
    for _ in range(len(datasets)):
        class_to_image_count.append({x: defaultdict(int) for x in all_classes})
    for _ in range(len(datasets)):
        image_count.append(defaultdict(int))
    for j in range(len(data)):
        cur_data = data[j]
        for i in range(len(cur_data)):
            image_count[j][cur_data[i]['image_id']] += 1
            identified_classes = []
            for cur_class in list(set(cur_data[i]['classes'])):
                identified_classes.append(cur_class)
                inner_cur_class = cur_class
                while inner_cur_class in child_to_parent:
                    inner_cur_class = child_to_parent[inner_cur_class]
                    identified_classes.append(inner_cur_class)
            identified_classes = list(set(identified_classes))
            for id_class in identified_classes:
                class_to_image_count[j][id_class][cur_data[i]['image_id']] += 1
        class_to_image_prob.append({x[0]: {y[0]: y[1]/image_count[j][y[0]] for y in x[1].items()} for x in class_to_image_count[j].items()})		

    return class_to_image_prob, class_to_image_count, image_count, image_ids

def get_annotator_agreement(dataset):
    all_classes = list(set(word_classes + list(parent_to_children.keys())))
    with open(f'datasets/{dataset}.json', 'r') as fp:
        data = json.load(fp)

    iid_to_captions = defaultdict(list)
    for x in data:
        iid_to_captions[x['image_id']].append(x)
        
    max_annotator_num = max([len(x) for x in iid_to_captions.values()])
    class_to_annotator_data = {x: np.zeros((len(iid_to_captions), max_annotator_num)) for x in all_classes}
    captions_grouped_by_iid = list(iid_to_captions.values())

    for i in range(len(captions_grouped_by_iid)):
        captions = captions_grouped_by_iid[i]
        for j in range(max_annotator_num):
            if j < len(captions):
                for cur_class in captions[j]['classes']:
                    class_to_annotator_data[cur_class][i, j] = 1
            else:
                for cur_class in all_classes:
                    class_to_annotator_data[cur_class][i, j] = np.nan

    class_to_agreement = {}
    for cur_class, data in class_to_annotator_data.items():
        if sum([len([j for j in range(data.shape[1]) if data[i][j] == 1]) for i in range(data.shape[0])]) == 0:
            continue
        df = pd.DataFrame(data)
        cac = CAC(df)
        class_to_agreement[cur_class] = cac.fleiss()['est']['coefficient_value']

    return class_to_agreement

def compute_wilcoxon(dataset_pairs):
    all_classes = list(set(word_classes + list(parent_to_children.keys())))

    res = []
    for dataset_pair in dataset_pairs:
        print(f'[Wilcoxon] starting {dataset_pair}')
        class_to_image_prob, _, _, image_ids = get_class_to_image_prob_dataset_pair(dataset_pair)
        res.append([])
        res[-1].append({})
        res[-1].append({})
        count = 0
        for cur_class in all_classes:
            print(f'\tStarting {cur_class}, {count} out of {len(all_classes)}', flush=True)
            count += 1
            res[-1][0][cur_class] = stats.wilcoxon([class_to_image_prob[0][cur_class][x] if x in class_to_image_prob[0][cur_class] else 0 for x in image_ids], [class_to_image_prob[1][cur_class][x] if x in class_to_image_prob[1][cur_class] else 0 for x in image_ids], alternative='greater', zero_method='zsplit').pvalue
            res[-1][1][cur_class] = stats.wilcoxon([class_to_image_prob[0][cur_class][x] if x in class_to_image_prob[0][cur_class] else 0 for x in image_ids], [class_to_image_prob[1][cur_class][x] if x in class_to_image_prob[1][cur_class] else 0 for x in image_ids], alternative='less', zero_method='zsplit').pvalue
    
    significant_classes = [all_classes, all_classes]
    for i in range(2):
        for cur_res in res:
            cur_significant_classes = [x[0] for x in cur_res[i].items() if x[1] < 0.05]
            significant_classes[i] = set(significant_classes[i]).intersection(cur_significant_classes)
    
    return significant_classes

dataset_to_image_num = {
    'COCO': 5,
    'STAIR-captions': 5,
    'YJCaptions': 5,
    'flickr30k': 5,
    'multi30k': 5,
    'coco-cn': 1,
    'flickr8kcn': 5
}
for dataset in all_datasets:
    if dataset.startswith('xm3600'):
        dataset_to_image_num[dataset] = 2

def get_extreme_images(dataset_pairs):
    res = []
    for dataset_pair in dataset_pairs:
        print(f'[extreme_images] starting {dataset_pair}')
        _, class_to_image_count, image_count, _ = get_class_to_image_prob_dataset_pair(dataset_pair)
        res.append([{}, {}])
        image_num_0 = dataset_to_image_num[dataset_pair[0]]
        image_num_1 = dataset_to_image_num[dataset_pair[1]]
        for cur_class in class_to_image_count[0].keys():
            res[-1][0][cur_class] = [x for x in image_count[0].keys() if image_count[0][x] >= image_num_0 and image_count[1][x] >= image_num_1 and class_to_image_count[0][cur_class][x] >= image_num_0 and class_to_image_count[1][cur_class][x] == 0]
            res[-1][1][cur_class] = [x for x in image_count[0].keys() if image_count[0][x] >= image_num_0 and image_count[1][x] >= image_num_1 and class_to_image_count[1][cur_class][x] >= image_num_1 and class_to_image_count[0][cur_class][x] == 0]

    return res

def compute_diff_and_add_indexes(dataset_pairs):
    all_classes = list(set(word_classes + list(parent_to_children.keys())))
    res = []
    for dataset_pair in dataset_pairs:
        print(f'[diff_add_index] starting {dataset_pair}')
        class_to_image_prob, _, _, image_ids = get_class_to_image_prob_dataset_pair(dataset_pair)
        diff_index = {}
        add_index = {}
        for cur_class in all_classes:
            image_set = [[x for x in image_ids if x in class_to_image_prob[i][cur_class] and class_to_image_prob[i][cur_class][x] >= 0.5] for i in range(2)]
            inter_set = set(image_set[0]).intersection(image_set[1])
            s1 = len(image_set[0])
            s2 = len(image_set[1])
            if s1 == 0 or s2 == 0:
                continue
            s_inter = len(inter_set)
            diff_index[cur_class] = s_inter/s1 - s_inter/s2
            add_index[cur_class] = s_inter/s1 + s_inter/s2
        res.append({'diff': diff_index, 'add': add_index})

    return res

def compute_correlation(dataset_pair):
    all_classes = list(set(word_classes + list(parent_to_children.keys())))
    print(f'[correlation] starting {dataset_pair}')
    class_to_image_prob, _, _, image_ids = get_class_to_image_prob_dataset_pair(dataset_pair)
    res = {}
    for cur_class in all_classes:
        lists = [[class_to_image_prob[i][cur_class][x] if x in class_to_image_prob[i][cur_class] else 0 for x in image_ids] for i in range(2)]
        res[cur_class] = stats.pearsonr(lists[0], lists[1])
    
    return res

def compute_vector_similarity(dataset_pair, sim_method):
    all_classes = list(set(word_classes + list(parent_to_children.keys())))
    class_to_image_prob, _, _, image_ids = get_class_to_image_prob_dataset_pair(dataset_pair)
    res = {}
    for cur_class in all_classes:
        lists = [[class_to_image_prob[i][cur_class][x] if x in class_to_image_prob[i][cur_class] else 0 for x in image_ids] for i in range(2)]
        vec1 = np.array(lists[0])
        vec2 = np.array(lists[1])
        if sim_method == 'l2_norm':
            res[cur_class] = (-1)*np.linalg.norm(vec1-vec2)
        elif sim_method == 'cosine':
            res[cur_class] = 1 - spatial.distance.cosine(vec1, vec2)
    return res

def compute_language_similarity(dataset_pair, sim_method, agg_method):
    per_class_similarity = compute_vector_similarity(dataset_pair, sim_method)
    sim_list = [x[1] for x in sorted(list(per_class_similarity.items()), key=lambda y:y[0])]
    if agg_method == 'mean':
        return sum(sim_list)/len(sim_list)
    if agg_method == 'l2_norm':
        return (-1)*np.linalg.norm(sim_list)
    
def cluster_langs(langs, sim_method, agg_method, n_cluster, dataset_prefix):
    pairs = [(i, j) for i in range(len(langs)) for j in range(i+1, len(langs))]
    adj_mat = np.zeros((36, 36))
    for i, j in tqdm(pairs):
        lang1 = langs[i]
        lang2 = langs[j]
        cur_adj = compute_language_similarity((f'{dataset_prefix}_{lang1}', f'{dataset_prefix}_{lang2}'), sim_method, agg_method)
        adj_mat[i, j] = cur_adj
        adj_mat[j, i] = cur_adj
        
    adj_mat = adj_mat-np.min(adj_mat)
    sc = SpectralClustering(n_cluster, affinity='precomputed', n_init=100)
    sc.fit(adj_mat)
    
    return [[langs[j] for j in range(len(langs)) if sc.labels_[j] == i] for i in range(n_cluster)]
