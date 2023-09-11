import stanza
from nltk.corpus import wordnet as wn

word_classes = [
    'man', 'woman', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'plant', 'bed', 'table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

nlp = stanza.Pipeline('en', tokenize_no_ssplit=True)

def get_depth_at_ind(token_list, i, depths):
    head_ind = token_list[i][0]['head'] - 1
    if head_ind == -1:
        depths[i] = 0
        return depths
    elif depths[head_ind] == -1:
        depths = get_depth_at_ind(token_list, head_ind, depths)
    depths[i] = depths[head_ind] + 1
    return depths

def get_depths(token_list):
    depths = [-1]*len(token_list)
    for i in range(len(token_list)):
        if depths[i] == -1:
            depths = get_depth_at_ind(token_list, i, depths)
    return depths

def find_word_classes(synset):
    word = synset.name().lower().split('.')[0]
    if word in word_classes:
        return [word]
    else:
        classes = []
        hypernyms = synset.hypernyms()
        for hypernym in hypernyms:
            classes += find_word_classes(hypernym)
        return list(set(classes))
    
def extract_noun_spans(caption):
    doc = nlp(caption)
    token_lists = [[x.to_dict() for x in y.tokens] for y in doc.sentences]
    if len(token_lists) > 1:
        return None
    token_list = token_lists[0]

    noun_spans = []

    # First find sequences of nouns
    noun_sequences = []
    in_sequence = False
    for i in range(len(token_list)):
        if token_list[i][0]['upos'] == 'NOUN' and (not in_sequence):
            sequence_start = i
            in_sequence = True
        if token_list[i][0]['upos'] != 'NOUN' and in_sequence:
            in_sequence = False
            noun_sequences.append((sequence_start, i))
    if in_sequence:
        noun_sequences.append((sequence_start, len(token_list)))

    # Next, for each sequence, find- for each token in the sequence- the highest ancestor inside the sequence
    for sequence_start, sequence_end in noun_sequences:
        highest_ancestors = []
        for token_ind in range(sequence_start, sequence_end):
            cur_ancestor = token_ind
            prev_cur_ancestor = cur_ancestor
            while cur_ancestor >= sequence_start and cur_ancestor < sequence_end:
                prev_cur_ancestor = cur_ancestor
                cur_ancestor = token_list[cur_ancestor][0]['head'] - 1
            highest_ancestors.append(prev_cur_ancestor)
        # A sequence of the same highest ancestor is a noun sequence
        noun_sequence_start = sequence_start
        cur_highest_ancestor = highest_ancestors[0]
        for i in range(1, len(highest_ancestors)):
            if highest_ancestors[i] != cur_highest_ancestor:
                noun_spans.append((noun_sequence_start, sequence_start + i))
                noun_sequence_start = sequence_start + i
                cur_highest_ancestor = highest_ancestors[i]
        noun_spans.append((noun_sequence_start, sequence_end))

    return noun_spans

def find_classes(caption):
    noun_spans = extract_noun_spans(caption)
    classes = []

    # TODO
    # for start_ind, end_ind in noun_spans:
    #     # Find the lowest noun
    #     lowest_noun_depth = max(depths) + 1
    #     lowest_noun_ind = -1
    #     for cur_ind in range(start_ind, end_ind):
    #         if depths[cur_ind] < lowest_noun_depth and token_list[cur_ind][0]['upos'] == 'NOUN':
    #             lowest_noun_depth = depths[cur_ind]
    #             lowest_noun_ind = cur_ind

    #     if lowest_noun_ind != -1:
    #         cur_word = token_list[lowest_noun_ind][0]['text']
    #         synsets = wn.synsets(cur_word)
    #         cur_classes = []
    #         for synset in synsets:
    #             cur_classes += find_word_classes(synset)
    #         if len(cur_classes) == 0:
    #             cur_class = None
    #         assert len(cur_classes) == 1, f'Word {cur_word} has multiple classes'
    #         cur_class = cur_classes[0]
    #     else:
    #         cur_class = None
    #     classes.append(cur_class)
    
    return classes
