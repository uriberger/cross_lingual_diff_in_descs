from find_classes_in_caption import *
from collections import defaultdict
import json
import scipy.stats as stats
from get_dataset import datasets as all_datasets

def get_class_to_image_prob(datasets):
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

def compute_wilcoxon(dataset_pairs):
    all_classes = list(set(word_classes + list(parent_to_children.keys())))

    res = []
    for dataset_pair in dataset_pairs:
        print(f'[Wilcoxon] starting {dataset_pair}')
        class_to_image_prob, _, image_ids, _ = get_class_to_image_prob(dataset_pair)
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
        _, class_to_image_count, image_count, _ = get_class_to_image_prob(dataset_pair)
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
        class_to_image_prob, _, _, image_ids = get_class_to_image_prob(dataset_pair)
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
    class_to_image_prob, _, _, _ = get_class_to_image_prob(dataset_pair)
    res = {}
    for cur_class in all_classes:
        image_ids = list(set(list(class_to_image_prob[0][cur_class].keys()) + list(class_to_image_prob[1][cur_class].keys())))
        if len(image_ids) < 2:
            continue
        lists = [[class_to_image_prob[i][cur_class][x] if x in class_to_image_prob[i][cur_class] else 0 for x in image_ids] for i in range(2)]
        res[cur_class] = stats.pearsonr(lists[0], lists[1])
    
    return res
