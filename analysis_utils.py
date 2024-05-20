from find_synsets_in_captions import *
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
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances as edist
import lang2vec.lang2vec as l2v
import mantel
import csv

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

def get_synset_to_image_prob_dataset_pair(datasets):
    with open(f'datasets/{datasets[0]}.json', 'r') as fp:
        data1 = json.load(fp)
    with open(f'datasets/{datasets[1]}.json', 'r') as fp:
        data2 = json.load(fp)
    image_ids = set([x['image_id'] for x in data1]).intersection([x['image_id'] for x in data2])
    data1 = [x for x in data1 if x['image_id'] in image_ids and x['synsets'] is not None]
    data2 = [x for x in data2 if x['image_id'] in image_ids and x['synsets'] is not None]
    data = [data1, data2]
    synset_to_image_count = []
    image_count = []
    synset_to_image_prob = []
    for _ in range(len(datasets)):
        synset_to_image_count.append({x: defaultdict(int) for x in all_synsets})
    for _ in range(len(datasets)):
        image_count.append(defaultdict(int))
    for j in range(len(data)):
        cur_data = data[j]
        for i in range(len(cur_data)):
            image_count[j][cur_data[i]['image_id']] += 1
            identified_synsets = []
            for synset in list(set([x[3] for x in cur_data[i]['synsets']])):
                identified_synsets.append(synset)
                inner_synset = synset
                while inner_synset in child2parent:
                    inner_synset = child2parent[inner_synset]
                    identified_synsets.append(inner_synset)
            identified_synsets = list(set(identified_synsets))
            for id_synset in identified_synsets:
                synset_to_image_count[j][id_synset][cur_data[i]['image_id']] += 1
        synset_to_image_prob.append({x[0]: {y[0]: y[1]/image_count[j][y[0]] for y in x[1].items()} for x in synset_to_image_count[j].items()})

    return synset_to_image_prob, synset_to_image_count, image_count, image_ids

def get_annotator_agreement(dataset):
    all_classes = list(set(word_classes2 + list(parent_to_children2.keys())))
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
                    class_to_annotator_data[cur_class[3]][i, j] = 1
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
    all_classes = list(set(word_classes2 + list(parent_to_children2.keys())))

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
    all_classes = list(set(word_classes2 + list(parent_to_children2.keys())))
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
    all_classes = list(set(word_classes2 + list(parent_to_children2.keys())))
    print(f'[correlation] starting {dataset_pair}')
    class_to_image_prob, _, _, image_ids = get_class_to_image_prob_dataset_pair(dataset_pair)
    res = {}
    for cur_class in all_classes:
        lists = [[class_to_image_prob[i][cur_class][x] if x in class_to_image_prob[i][cur_class] else 0 for x in image_ids] for i in range(2)]
        res[cur_class] = stats.pearsonr(lists[0], lists[1])
    
    return res

def compute_similarity_by_synset(dataset_pair, sim_method, synset):
    synset_to_image_prob, _, _, image_ids = get_synset_to_image_prob_dataset_pair(dataset_pair)
    lists = [[synset_to_image_prob[i][synset][x] if x in synset_to_image_prob[i][synset] else 0 for x in image_ids] for i in range(2)]
    vec1 = np.array(lists[0])
    vec2 = np.array(lists[1])
    if sim_method == 'l2_norm':
        return (-1)*np.linalg.norm(vec1-vec2)
    elif sim_method == 'cosine':
        return 1 - spatial.distance.cosine(vec1, vec2)

def compute_vector_similarity(dataset_pair, sim_method):
    synset_to_image_prob, _, _, image_ids = get_synset_to_image_prob_dataset_pair(dataset_pair)
    res = {}
    for synset in all_synsets:
        lists = [[synset_to_image_prob[i][synset][x] if x in synset_to_image_prob[i][synset] else 0 for x in image_ids] for i in range(2)]
        vec1 = np.array(lists[0])
        vec2 = np.array(lists[1])
        if sim_method == 'l2_norm':
            res[synset] = (-1)*np.linalg.norm(vec1-vec2)
        elif sim_method == 'cosine':
            res[synset] = 1 - spatial.distance.cosine(vec1, vec2)
    return res

def compute_language_similarity(dataset_pair, sim_method, agg_method, synset):
    if synset is None:
        per_synset_similarity = compute_vector_similarity(dataset_pair, sim_method)
        # Use all synsets an aggregeate using agg_method
        sim_list = [x[1] for x in sorted(list(per_synset_similarity.items()), key=lambda y:y[0])]
        if agg_method == 'mean':
            return sum(sim_list)/len(sim_list)
        if agg_method == 'l2_norm':
            return (-1)*np.linalg.norm(sim_list)
    else:
        return compute_similarity_by_synset(dataset_pair, sim_method, synset)
    
def cluster_langs(langs, sim_method, agg_method, n_cluster, dataset_prefix, synset=None):
    pairs = [(i, j) for i in range(len(langs)) for j in range(i+1, len(langs))]
    adj_mat = np.zeros((36, 36))
    for i, j in tqdm(pairs):
        lang1 = langs[i]
        lang2 = langs[j]
        cur_adj = compute_language_similarity((f'{dataset_prefix}_{lang1}', f'{dataset_prefix}_{lang2}'), sim_method, agg_method, synset)
        adj_mat[i, j] = cur_adj
        adj_mat[j, i] = cur_adj
        
    adj_mat = adj_mat-np.min(adj_mat)
    sc = SpectralClustering(n_cluster, affinity='precomputed', n_init=100)
    sc.fit(adj_mat)
    
    return [[langs[j] for j in range(len(langs)) if sc.labels_[j] == i] for i in range(n_cluster)]

def plot_similarity_heatmap(langs):
    sim_method = 'l2_norm'
    agg_method = 'l2_norm'
    synset = None
    dataset_prefix = 'xm3600'

    pairs = [(i, j) for i in range(len(langs)) for j in range(i+1, len(langs))]
    adj_mat = np.zeros((36, 36))
    for i, j in tqdm(pairs):
        lang1 = langs[i]
        lang2 = langs[j]
        cur_adj = compute_language_similarity((f'{dataset_prefix}_{lang1}', f'{dataset_prefix}_{lang2}'), sim_method, agg_method, synset)
        adj_mat[i, j] = cur_adj
        adj_mat[j, i] = cur_adj

    plt.clf()
    plt.figure(figsize=(12,9))
    my_langs = ['cs', 'da', 'de', 'el', 'es', 'fi', 'fr', 'hr', 'hu', 'it', 'nl', 'no', 'pl', 'pt', 'sv', 'id', 'ja', 'ko', 'th', 'vi', 'zh', 'ru', 'uk', 'ar', 'en', 'fa', 'fil', 'hi', 'quz', 'sw', 'te', 'tr', 'he', 'ro', 'bn', 'mi']
    adj_mat2 = np.zeros((36, 36))
    l2i = {langs[i]: i for i in range(36)}
    for i in range(36):
        for j in range(36):
            adj_mat2[35-i, j] = adj_mat[l2i[my_langs[i]], l2i[my_langs[j]]]
    for i in range(36):
        adj_mat2[35-i,i] = adj_mat2[0,1]
    for i in range(36):
        adj_mat2[35-i,i] = np.max(adj_mat2)
    plt.xticks(ticks=[x+0.5 for x in range(36)], labels=my_langs)
    plt.yticks(ticks=[x+0.5 for x in range(36)], labels=[my_langs[35-i] for i in range(36)])
    plt.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.tick_params(axis='both', labelsize=10)
    heatmap = plt.pcolor(adj_mat2, cmap='hot')
    plt.colorbar(heatmap)
    plt.savefig('del_me.png')

def get_object_num_by_location(langs, synset):
    with open('/cs/labs/oabend/uriber/datasets/crossmodal3600/captions.jsonl', 'r') as fp:
        jl = list(fp)
    data = [json.loads(x) for x in jl]
    iid2l = {int(x['image/key'], 16): x['image/locale'] for x in data}
    image_num = 100
    l2iid2count = {lang: defaultdict(list) for lang in langs}
    
    for lang in langs:
        with open(f'datasets/xm3600_{lang}.json', 'r') as fp:
            data = json.load(fp)
        for sample in data:
            if synset is None:
                l2iid2count[iid2l[sample['image_id']]][sample['image_id']].append(len(sample['synsets']))
            else:
                l2iid2count[iid2l[sample['image_id']]][sample['image_id']].append(len([x for x in sample['synsets'] if is_hyponym_of(x[3], synset)]))
    
    l2count = {x[0]: sum([sum(y)/len(y) for y in x[1].values()]) for x in l2iid2count.items()}
    res = sorted([(x[0], '%.2f' % (x[1]/image_num)) for x in l2count.items()], key=lambda x:x[1])
    
    return res

def get_object_num_by_language(langs, synset):
    image_num = 3600
    l2iid2count = {lang: defaultdict(list) for lang in langs}
    
    for lang in langs:
        with open(f'datasets/xm3600_{lang}.json', 'r') as fp:
            data = json.load(fp)
        for sample in data:
            if synset is None:
                l2iid2count[lang][sample['image_id']].append(len(sample['synsets']))
            else:
                l2iid2count[lang][sample['image_id']].append(len([x for x in sample['synsets'] if is_hyponym_of(x[3], synset)]))
    
    l2count = {x[0]: sum([sum(y)/len(y) for y in x[1].values()]) for x in l2iid2count.items()}
    res = sorted([(x[0], '%.2f' % (x[1]/image_num)) for x in l2count.items()], key=lambda x:x[1])
    
    return res

def plot_object_num(langs, synset_list, by_location):
    assert len(synset_list) in [1, 3]
    
    plt.clf()
    if len(synset_list) == 1:
        row_num = 1
    else:
        row_num = 2
    col_num = 2
    fig, axs = plt.subplots(row_num, col_num)
    fig.set_size_inches(12,8*row_num)
    
    if by_location:
        overall_res = get_object_num_by_location(langs, None)
    else:
        overall_res = get_object_num_by_language(langs, None)
    
    if len(synset_list) == 1:
        axs[0].barh(range(36), width=[float(x[1]) for x in overall_res])
        axs[0].set_title('Overall')
        axs[0].set_yticks(ticks=range(36), labels=[x[0] for x in overall_res])
        axs[0].tick_params(axis='both', labelsize=10)
    else:
        axs[0, 0].barh(range(36), width=[float(x[1]) for x in overall_res])
        axs[0, 0].set_title('Overall')
        axs[0, 0].set_yticks(ticks=range(36), labels=[x[0] for x in overall_res])
        axs[0, 0].tick_params(axis='both', labelsize=10)
    row = 0
    col = 1
    for i in range(len(synset_list)):
        if by_location:
            res = get_object_num_by_location(langs, synset_list[i])
        else:
            res = get_object_num_by_language(langs, synset_list[i])
        if len(synset_list) == 1:
            axs[col].barh(range(36), width=[float(x[1]) for x in res])
            axs[col].set_title(synset_list[i])
            axs[col].set_yticks(ticks=range(36), labels=[x[0] for x in res])
            axs[col].tick_params(axis='both', labelsize=10)
        else:
            axs[row, col].barh(range(36), width=[float(x[1]) for x in res])
            axs[row, col].set_title(synset_list[i])
            axs[row, col].set_yticks(ticks=range(36), labels=[x[0] for x in res])
            axs[row, col].tick_params(axis='both', labelsize=10)
        col += 1
        if col == col_num:
            col = 0
            row += 1
    
    plt.savefig('del_me.png')

def plot_legend():
    colors = ['black', 'red', 'green', 'blue', 'orange', 'yellow', 'gray']
	
    plt.clf()
    row_num = 1
    col_num = 3
    fig, axs = plt.subplots(row_num, col_num)
    fig.set_size_inches(40,15)
    
    fig.legend(handles=[mlines.Line2D([], [], color=colors[i], marker='s', ls='', label=f'Cluster {i+1}') for i in range(len(colors))], fontsize=25, loc='lower center', ncols=4)
    
    plt.savefig('del_me.png')

def get_lang_synset_image_matrix(root_only):
    labels = [x.split('_')[1] for x in all_datasets if x.startswith('xm3600_')]
    if root_only:
        concepts = [x for x in all_synsets if x not in child2parent]
    else:
        concepts = list(all_synsets)
    X = np.zeros((len(labels), len(concepts), 3600))

    with open(f'datasets/xm3600_en.json', 'r') as fp:
        data = json.load(fp)
    all_images = list(set([x['image_id'] for x in data]))
    for i, lang in tqdm(enumerate(labels)):
        synset2prob = get_synset_to_image_prob(f'xm3600_{lang}')[0]
        for j, synset in enumerate(concepts):
            for k, image_id in enumerate(all_images):
                if image_id in synset2prob[synset]:
                    X[i, j, k] = synset2prob[synset][image_id]

    return labels, concepts, X

def plot_saliency_heatmap(sort_by_mean):
    sns.set_style('whitegrid')
    plt.rcParams["font.family"] = "Times New Roman"

    labels, concepts, X = get_lang_synset_image_matrix(True)

    X = X.sum(axis=-1)
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = np.std(X, axis=0)
    Z = (X - X_mean) / X_std
    if sort_by_mean:
        Z = np.concatenate((Z, Z.mean(axis=1)[:, np.newaxis]), axis=1)
        concepts.append('Mean')
        lang_ind_mean_list = [(i, Z[i, -1]) for i in range(Z.shape[0])]
        lang_ind_mean_list.sort(key=lambda x:x[1])
        permutation = [[j for j in range(len(labels)) if lang_ind_mean_list[j][0] == i][0] for i in range(len(labels))]
        idx = np.empty_like(permutation)
        idx[permutation] = np.arange(len(permutation))
        Z[:] = Z[idx, :]
        labels = [labels[lang_ind_mean_list[i][0]] for i in range(len(labels))]
    X = pd.DataFrame(Z.T, columns=labels, index=concepts)

    ax = sns.heatmap(X, cmap="vlag",
                     center=0, annot=True, fmt=".1f", square=True, xticklabels=True, yticklabels=True, annot_kws={"fontsize":5})
    plt.savefig('saliency_heatmap.pdf', bbox_inches='tight')

def run_saliency_similarity_correlation_test():
    labels, _, X = get_lang_synset_image_matrix(False)
    X = np.reshape(X, (X.shape[0], -1))
    X_sim = edist(X)

    iso_639_2_to_639_3 = {
        'ar': 'ara',
        'bn': 'ben',
        'cs': 'ces',
        'da': 'dan',
        'de': 'deu',
        'el': 'ell',
        'en': 'eng',
        'es': 'spa',
        'fa': 'fas',
        'fi': 'fin',
        'fil': 'fil',
        'fr': 'fra',
        'he': 'heb',
        'hi': 'hin',
        'hr': 'hrv',
        'hu': 'hun',
        'id': 'ind',
        'it': 'ita',
        'ja': 'jpn',
        'ko': 'kor',
        'mi': 'mri',
        'nl': 'nld',
        'no': 'nor',
        'pl': 'pol',
        'pt': 'por',
        'quz': 'quz',
        'ro': 'ron',
        'ru': 'rus',
        'sv': 'swe',
        'sw': 'swa',
        'te': 'tel',
        'th': 'tha',
        'tr': 'tur',
        'uk': 'ukr',
        'vi': 'vie',
        'zh': 'zho'
    }

    labels = [iso_639_2_to_639_3[code] for code in labels]
    for criterion in ["geographic", "genetic", "featural"]:
        d = l2v.distance(criterion, labels)
        res = mantel.test(X_sim, d)
        print(criterion, res)

def get_vertical_depth(synset):
	hypernyms = synset.hypernyms() + synset.instance_hypernyms()
	if len(hypernyms) == 0:
		return 0
	return min([get_vertical_depth(x) for x in hypernyms]) + 1

def get_lang_to_gran_list(langs, root_synset=None):
    iid2root_synset = get_image_id_to_root_synsets()

    l2gran = {}
    for lang in tqdm(langs):
        with open(f'datasets/xm3600_{lang}.json', 'r') as fp:
            data = json.load(fp)
        l2gran[lang] = []
        for sample in data:
            for synset in sample['synsets']:
                if (root_synset is not None) and (not is_hyponym_of(synset[3], root_synset)):
                    continue
                if not verify_synset_in_image(synset, sample['image_id'], iid2root_synset):
                    continue
                l2gran[lang].append(get_vertical_depth(wn.synset(synset[3])) + synset[4])
	
    return l2gran

def granularity_analysis():
    sns.set_style('whitegrid')
    plt.rcParams["font.family"] = "Times New Roman"

    langs = [x.split('_')[1] for x in all_datasets if x.startswith('xm3600_')]
    l2gran = get_lang_to_gran_list(langs)
    samples = [d for _, d in l2gran.items()]
    res = stats.kruskal(*samples)
    print(res)

    langs_list = [l for l, d in l2gran.items() for _ in d]
    depths = [x for _, d in l2gran.items() for x in d]

    df = pd.DataFrame.from_dict({
            "language": langs_list,
            "depth": depths
        })

    ax = sns.histplot(df, x="depth", hue="language", common_norm=False, 
        binwidth=1, stat="probability", element="step", alpha=0.01)
    ax.set_xticks(range(18))

    sns.move_legend(
        ax, "upper left", bbox_to_anchor=(1, 1), ncol=2, title=None, frameon=True,
    )
    plt.savefig('depths_dist.pdf', bbox_inches='tight')
