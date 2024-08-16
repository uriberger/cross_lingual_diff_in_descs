from collections import defaultdict
import json
import scipy.stats as stats
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances as edist
import lang2vec.lang2vec as l2v
import mantel
import statistics
from nltk.corpus import wordnet as wn

from utils import get_image_id_to_root_synsets,\
    verify_synset_in_image,\
    is_hyponym_of,\
    langs,\
    east_asian_langs,\
    low_resource_langs,\
    child2parent,\
    all_synsets
from config import xm3600_json_path, use_low_resource_langs
from get_dataset import get_processed_dataset

def get_synset_to_image_prob(dataset):
    iid2root_synset = get_image_id_to_root_synsets()

    data = get_processed_dataset(dataset)
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
        if sample['image_id'] in iid2root_synset:
            identified_synsets = [synset for synset in identified_synsets if verify_synset_in_image(synset, sample['image_id'], iid2root_synset)]
        for id_synset in identified_synsets:
            synset_to_image_count[id_synset][sample['image_id']] += 1
    synset_to_image_prob = {x[0]: {y[0]: y[1]/image_count[y[0]] for y in x[1].items()} for x in synset_to_image_count.items()}

    return synset_to_image_prob, synset_to_image_count, image_count
    
def get_object_num_by_location(synset):
    iid2root_synset = get_image_id_to_root_synsets()
    with open(xm3600_json_path, 'r') as fp:
        jl = list(fp)
    data = [json.loads(x) for x in jl]
    if not use_low_resource_langs:
        data = [x for x in data if x['image/locale'] not in low_resource_langs]
    iid2l = {int(x['image/key'], 16): x['image/locale'] for x in data}
    l2iid2count = {lang: defaultdict(list) for lang in langs}
    
    for lang in langs:
        data = get_processed_dataset(f'xm3600_{lang}')
        for sample in data:
            if sample['image_id'] not in iid2l:
                continue
            if synset is None:
                l2iid2count[iid2l[sample['image_id']]][sample['image_id']].append(len([x for x in sample['synsets'] if verify_synset_in_image(x[3], sample['image_id'], iid2root_synset)]))
            else:
                l2iid2count[iid2l[sample['image_id']]][sample['image_id']].append(len([x for x in sample['synsets'] if is_hyponym_of(x[3], synset) and verify_synset_in_image(x[3], sample['image_id'], iid2root_synset)]))
    
    return l2iid2count

def get_object_num_by_language(synset):
    iid2root_synset = get_image_id_to_root_synsets()
    l2iid2count = {lang: defaultdict(list) for lang in langs}
    
    for lang in langs:
        data = get_processed_dataset(f'xm3600_{lang}')
        for sample in data:
            if synset is None:
                l2iid2count[lang][sample['image_id']].append(len([x for x in sample['synsets'] if verify_synset_in_image(x[3], sample['image_id'], iid2root_synset)]))
            else:
                l2iid2count[lang][sample['image_id']].append(len([x for x in sample['synsets'] if is_hyponym_of(x[3], synset) and verify_synset_in_image(x[3], sample['image_id'], iid2root_synset)]))
    
    return l2iid2count

def run_object_num_by_location_analysis():
    l2iid2count = get_object_num_by_location(None)
    l2iid2mean = {x[0]: {y[0]: sum(y[1])/len(y[1]) for y in x[1].items()} for x in l2iid2count.items()}

    ea_list = []
    for x in l2iid2mean.items():
        if x[0] in east_asian_langs:
            ea_list += list(x[1].values())

    other_list = []
    for x in l2iid2mean.items():
        if x[0] not in east_asian_langs:
            other_list += list(x[1].values())

    return stats.mannwhitneyu(ea_list, other_list, alternative='greater')
	
def run_object_num_by_language_analysis():
    l2iid2count = get_object_num_by_language(None)
    l2iid2mean = {x[0]: {y[0]: sum(y[1])/len(y[1]) for y in x[1].items()} for x in l2iid2count.items()}

    en_data = get_processed_dataset('xm3600_en')
    iids = list(set([x['image_id'] for x in en_data]))

    ea_iid2mean = {iid: sum([l2iid2mean[lang][iid] for lang in east_asian_langs])/len(east_asian_langs) for iid in iids}
    other_langs = [x for x in langs if x not in east_asian_langs]
    other_iid2mean = {iid: sum([l2iid2mean[lang][iid] for lang in other_langs])/len(other_langs) for iid in iids}

    ea_list = sorted(list(ea_iid2mean.items()), key=lambda x:x[0])
    ea_list = [x[1] for x in ea_list]
    other_list = sorted(list(other_iid2mean.items()), key=lambda x:x[0])
    other_list = [x[1] for x in other_list]

    return stats.wilcoxon(ea_list, other_list, zero_method='wilcox', alternative='greater')

def plot_object_num(by_location):
    plt.clf()
    plt.figure().set_figwidth(3)
    
    if by_location:
        l2iid2count = get_object_num_by_location(None)
        image_num = 100
    else:
        l2iid2count = get_object_num_by_language(None)
        image_num = 3600
    l2count = {x[0]: sum([sum(y)/len(y) for y in x[1].values()]) for x in l2iid2count.items()}
    res = sorted([(x[0], '%.2f' % (x[1]/image_num)) for x in l2count.items()], key=lambda x:x[1])
    
    ylabels_size = 8

    plt.barh(range(len(langs)), width=[float(x[1]) for x in res], color=['red' if x[0] in east_asian_langs else 'black' for x in res])
    plt.yticks(ticks=range(len(langs)), labels=[x[0] for x in res], size=ylabels_size, family='Times New Roman')
    plt.tick_params(axis='x', labelfontfamily='Times New Roman')
    
    by_str = 'location' if by_location else 'language'
    plt.savefig(f'object_num_by_{by_str}.png', dpi=200)

def get_lang_synset_image_matrix(root_only):
    if root_only:
        concepts = [x for x in all_synsets if x not in child2parent]
    else:
        concepts = list(all_synsets)
    X = np.zeros((len(langs), len(concepts), 3600))

    data = get_processed_dataset('xm3600_en')
    all_images = list(set([x['image_id'] for x in data]))
    for i, lang in tqdm(enumerate(langs)):
        synset2prob = get_synset_to_image_prob(f'xm3600_{lang}')[0]
        for j, synset in enumerate(concepts):
            for k, image_id in enumerate(all_images):
                if image_id in synset2prob[synset]:
                    X[i, j, k] = synset2prob[synset][image_id]

    return langs, concepts, X

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

def get_lang_to_gran_list(root_synset=None):
    iid2root_synset = get_image_id_to_root_synsets()

    l2gran = {}
    for lang in tqdm(langs):
        data = get_processed_dataset(f'xm3600_{lang}')
        l2gran[lang] = []
        for sample in data:
            for synset in sample['synsets']:
                if (root_synset is not None) and (not is_hyponym_of(synset[3], root_synset)):
                    continue
                if not verify_synset_in_image(synset[3], sample['image_id'], iid2root_synset):
                    continue
                l2gran[lang].append(get_vertical_depth(wn.synset(synset[3])) + synset[4])
	
    return l2gran

def granularity_analysis():
    sns.set_style('whitegrid')
    plt.rcParams["font.family"] = "Times New Roman"

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
        ax, "upper left", bbox_to_anchor=(0, 1.4), ncol=6, title=None, frameon=True,
    )
    plt.savefig('depths_dist.pdf', bbox_inches='tight')

def synset_agreement_analysis():
    # Analyze on which synsets annotators from different languages tend to agree how salient they are
    root_synsets = set([x for x in all_synsets if x not in child2parent])
    iid2root_synset = get_image_id_to_root_synsets()

    # First, for each synset, compute the standard deviation of saliency across langauges in each image, and average over all images
    iid2probs = {}
    for lang in langs:
        synset2prob, _, image_count = get_synset_to_image_prob(f'xm3600_{lang}')
        image_ids = image_count.keys()
        for image_id in image_ids:
            for synset, synset_data in synset2prob.items():
                if synset not in root_synsets:
                    continue
                if image_id not in iid2root_synset:
                    continue
                if synset not in iid2root_synset[image_id]:
                    continue
                if not image_id in iid2probs:
                    iid2probs[image_id] = {}
                if not synset in iid2probs[image_id]:
                    iid2probs[image_id][synset] = []
                if image_id in synset_data:
                    iid2probs[image_id][synset].append(synset_data[image_id])
                else:
                    iid2probs[image_id][synset].append(0)
    iid2std = {}
    for image_id, synset2probs in iid2probs.items():
        iid2std[image_id] = {}
        for synset, prob_list in synset2probs.items():
            if sum(prob_list) > 0:
                iid2std[image_id][synset] = statistics.stdev(prob_list)
    synset2stds = defaultdict(list)
    for image_id, image_stds in iid2std.items():
        for synset, std in image_stds.items():
            synset2stds[synset].append(std)
    synset2mean_std = {x[0]: statistics.mean(x[1]) for x in synset2stds.items()}
    mean_of_stds = sorted(list(synset2mean_std.items()), key=lambda x:x[1])
    print('Mean of stds')
    print(mean_of_stds)
	
    # Next, for each synset, compute the mean saliency over all images, and compute the standard deviation across langauges
    synset2count = defaultdict(int)
    for synset_list in iid2root_synset.values():
        for synset in synset_list:
            synset2count[synset] += 1
    lang2probs = {}
    for lang in tqdm(langs):
        synset2prob, _, _ = get_synset_to_image_prob(f'xm3600_{lang}')
        lang2probs[lang] = {synset: sum([x[1] for x in synset2prob[synset].items() if x[0] in iid2root_synset and synset in iid2root_synset[x[0]]])/synset2count[synset] for synset in root_synsets}
    synset2means = defaultdict(list)
    for lang, synsets_data in lang2probs.items():
        for synset, mean_prob in synsets_data.items():
            synset2means[synset].append((lang, mean_prob))
    synset2std_of_means = {x[0]: statistics.stdev([y[1] for y in x[1]]) for x in synset2means.items()}
    std_of_means = sorted(list(synset2std_of_means.items()), key=lambda x:x[1])
    print('Std of means:')
    print(std_of_means)
	
    # Now, compute correlation with how common each synset it
    counts = [x[1] for x in sorted(list(synset2count.items()), key=lambda y:y[0])]
    means = [x[1] for x in sorted(mean_of_stds, key=lambda y:y[0])]
    stds = [x[1] for x in sorted(std_of_means, key=lambda y:y[0])]
    print('Pearson correlation with means:')
    print(stats.pearsonr(counts, means))
    print('Pearson correlation with stds:')
    print(stats.pearsonr(counts, stds))
