import stanza
from nltk.corpus import wordnet as wn

word_classes = [
    'man', 'woman', 'boy', 'girl', 'person', 'people', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'sign', 'parking meter', 'bench', 'bird', 
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'plate', 'bottle', 'glass', 'cup', 'can',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'corn',
    'vegetable', 'fruit', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'plant', 'bed', 'table',
    'toilet', 'television', 'laptop', 'computer', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush', 'wall', 'sidewalk', 'mountain', 'beach', 'kitchen', 'kitchen utensil', 'graffiti',
    'tree', 'sky', 'sun', 'moon', 'camera', 'mirror', 'teeth', 'bathtub', 'wine', 'sea', 'lake'
    ]

known_mappings = {
    'rail road track': 'railroad track', 'tv': 'television', 'skate board': 'skateboard', 'cats': 'cat',
    'snowboarder': 'person', 'surfer': 'person', 'ocean': 'sea', 'remote-control': 'remote'
}

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

def get_synset_count(synset):
    count = 0
    for lemma in synset.lemmas():
        count += lemma.count()
    return count

def find_synset_classes(synset):
    word = synset.name().lower().split('.')[0]
    if word in word_classes:
        return [word]
    else:
        classes = []
        hypernyms = synset.hypernyms()
        for hypernym in hypernyms:
            classes += find_synset_classes(hypernym)
        return list(set(classes))

def find_phrase_class(phrase):
    if phrase in known_mappings:
        phrase = known_mappings[phrase]
    if phrase in word_classes:
        phrase_class = phrase
    else:
        synsets = wn.synsets(phrase)
        classes = []
        all_synsets_count = sum([get_synset_count(x) for x in synsets])
        for synset in synsets:
            if all_synsets_count == 0 or get_synset_count(synset)/all_synsets_count > 0.2:
                classes += find_synset_classes(synset)
        classes = list(set(classes))
        if len(classes) == 0:
            return None
        else:
            assert len(classes) == 1, f'Phrase "{phrase}" has multiple classes'
            phrase_class = classes[0]
    return phrase_class
    
def extract_noun_spans(token_list):
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
                noun_spans.append((noun_sequence_start, sequence_start + i, cur_highest_ancestor))
                noun_sequence_start = sequence_start + i
                cur_highest_ancestor = highest_ancestors[i]
        noun_spans.append((noun_sequence_start, sequence_end, cur_highest_ancestor))

    return noun_spans

def preprocess(caption):
    # Just solving some known issues

    # 1. Every time we have 'remote control' in a sentence, 'remote' is an adjective so the identified noun span is
    # 'control', which isn't what we want. So we'll change it to 'remote'
    remote_control_start = caption.find('remote control')
    if remote_control_start != -1:
        caption = caption[:remote_control_start] + 'remote-control' + caption[remote_control_start+14:]

    return caption

def find_classes(caption):
    caption = preprocess(caption)
    doc = nlp(caption)
    token_lists = [[x.to_dict() for x in y.tokens] for y in doc.sentences]
    if len(token_lists) > 1:
        return None
    token_list = token_lists[0]

    noun_spans = extract_noun_spans(token_list)
    classes = []

    for start_ind, end_ind, highest_ancestor_ind in noun_spans:
        phrase = ' '.join([token_list[i][0]['text'] for i in range(start_ind, end_ind)]).lower()
        phrase_class = find_phrase_class(phrase)
        if phrase_class is None:
            phrase = token_list[highest_ancestor_ind][0]['text']
            phrase_class = find_phrase_class(phrase)
        classes.append((start_ind, end_ind, phrase_class))
    
    return classes
