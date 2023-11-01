import stanza
from nltk.corpus import wordnet as wn
import inflect
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from torch import nn
import math

word_classes = [
    'man', 'woman', 'boy', 'girl', 'child', 'person', 'people', 'bicycle', 'car', 'motorcycle', 'airplane', 'blimp', 'bus',
    'train', 'truck', 'boat', 'ship', 'watercraft', 'traffic light', 'fire hydrant', 'sign', 'parking meter', 'bench',
    'bird', 'ostrich', 'wasp', 'fish', 'tuna', 'cat', 'dog', 'horse', 'fox', 'sheep', 'cow', 'elephant', 'bear', 'tiger',
    'chicken', 'zebra', 'giraffe', 'groundhog', 'pig', 'deer', 'gazelle', 'goose', 'shrimp', 'worm', 'turtle', 'bunny',
    'chameleon', 'rat', 'insect', 'beetle', 'butterfly', 'spider', 'weasel', 'wolverine', 'animal', 'beaver', 'badger',
    'backpack', 'umbrella', 'tie', 'hat', 'sunglasses', 'shirt', 'sweater', 'pants', 'diaper', 'dress', 'coat', 'boa',
    'clothing', 'suitcase', 'frisbee', 'ski', 'snowboard', 'ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'plate', 'bottle', 'glass', 'cup', 'can', 'fork', 'knife', 'spoon', 'bowl', 'tray',
    'banana', 'apple', 'kiwi', 'raspberry', 'sandwich', 'orange', 'mandarin', 'cucumber', 'tomato', 'chickpea', 'broccoli',
    'brussel sprout', 'carrot', 'corn', 'garlic', 'onion', 'soybean', 'sausage', 'cabbage', 'vegetable', 'fruit', 'hotdog',
    'pizza', 'fries', 'donut', 'cake', 'biscuit', 'burrito', 'bread', 'toast', 'coffee', 'chair', 'couch', 'plant', 'bed',
    'pillow', 'blanket', 'sheets', 'mattress', 'table', 'counter', 'toilet', 'television', 'laptop', 'computer', 'monitor',
    'mouse', 'remote', 'controller', 'keyboard', 'phone', 'microwave', 'oven', 'stove', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'doll', 'hair drier', 'toothbrush', 'wall', 'door', 'windows',
    'sidewalk', 'building', 'restaurant', 'mountain', 'beach', 'kitchen', 'kitchen utensil', 'graffiti', 'tree', 'sky',
    'sun', 'moon', 'camera', 'mirror', 'teeth', 'bathtub', 'wine', 'sea', 'lake', 'mouth', 'ear', 'eye', 'nose',
    'platform', 'box', 'uniform', 'towel', 'stone', 'statue', 'candle', 'rope', 'nut', 'bag', 'pole', 'toothpick', 'wheel',
    'basket', 'nail', 'hammer', 'shovel', 'hand tool', 'guitar', 'piano', 'musical instrument', 'newspaper', 'helmet',
    'carrier', 'slicer', 'cutter', 'caboose', 'pinwheel', 'fireball', 'okra', 'siren', 'pen', 'pencil', 'shingle'
    ]

parent_to_children = {
    'person': ['man', 'woman', 'boy', 'girl', 'child', 'people'],
    'vehicle': ['bicycle', 'car', 'motorcycle', 'aircraft', 'bus', 'train', 'watercraft'],
    'car': ['truck'],
    'watercraft': ['boat', 'ship'],
    'aircraft': ['airplane', 'blimp'],
    'seat': ['bench', 'chair', 'couch'],
    'furniture': ['bed', 'seat', 'table', 'counter'],
    'bedding accessories': ['pillow', 'blanket', 'sheets', 'mattress'],
    'animal': ['bird', 'fish', 'mammal', 'goose', 'shrimp', 'worm', 'turtle', 'chicken', 'rat', 'insect', 'spider',
               'chameleon'],
    'mammal': ['cat', 'dog', 'horse', 'sheep', 'cow', 'wild mammal', 'groundhog', 'pig', 'deer', 'gazelle', 'bunny',
               'beaver', 'fox', 'weasel', 'badger'],
    'fish': ['tuna'],
    'insect': ['wasp', 'beetle', 'butterfly'],
    'wild mammal': ['elephant', 'bear', 'zebra', 'giraffe', 'tiger', 'wolverine'],
    'bag': ['backpack', 'suitcase', 'basket'],
    'clothing': ['tie', 'hat', 'sunglasses', 'shirt', 'sweater', 'pants', 'diaper', 'dress', 'coat', 'helmet', 'boa'],
    'riding device': ['skis', 'surfboard', 'snowboard', 'skateboard'],
    'game': ['frisbee', 'sport instrument', 'kite'],
    'sport instrument': ['ball', 'baseball bat', 'baseball glove', 'tennis racket'],
    'kitchen utensil': ['plate', 'cup', 'can', 'fork', 'knife', 'spoon', 'bowl', 'tray'],
    'cup': ['glass'],
    'food': ['fruit', 'vegetable', 'sandwich', 'corn', 'sausage', 'hotdog', 'pizza', 'fries', 'donut', 'cake', 'burrito',
             'bread', 'toast', 'biscuit'],
    'fruit': ['banana', 'apple', 'orange', 'mandarin', 'kiwi', 'raspberry', 'nut'],
    'vegetable': ['cucumber', 'tomato', 'broccoli', 'brussel sprout', 'carrot', 'garlic', 'onion', 'cabbage', 'chickpea',
                  'okra', 'soybean'],
    'plant': ['tree'],
    'electornics': ['television', 'laptop', 'computer', 'monitor', 'mouse', 'remote', 'controller', 'keyboard', 'phone',
    'microwave', 'oven', 'stove', 'toaster', 'refrigerator'],
    'body part': ['mouth', 'ear', 'eye', 'nose'],
    'hand tool': ['hammer', 'shovel'],
    'musical instrument': ['guitar', 'piano']
}

child_to_parent = {}
for parent, children in parent_to_children.items():
    for child in children:
        child_to_parent[child] = parent

def is_hyponym_of(class1, class2):
    if class1 == class2:
        return True
    while class1 in child_to_parent:
        return is_hyponym_of(child_to_parent[class1], class2)
    return False

non_word_classes = [
    'sport', 'amazon', 'quarry', 'aa', 'cob'
]

known_mappings = {
    'rail road track': 'railroad track', 'tv': 'television', 'skate board': 'skateboard', 'roller blades': 'rollerblade',
    'snowboarder': 'person', 'surfer': 'person', 'ocean': 'sea', 'remote-control': 'remote', 'scooter': 'motorcycle',
    'hay': 'plant', 'van': 'car', 'walnut': 'nut', 'peanut': 'nut', 'children': 'child', 'diner': 'restaurant',
    'guy': 'man', 'tennis racquet': 'tennis racket', 'male': 'man', 'female': 'woman', 'adult': 'person',
    'plantain': 'banana', 'racer': 'person', 'young': 'person', 'clippers': 'scissors', 'pet': 'animal',
    'president': 'person', 'guide': 'person', 'climber': 'person', 'commuter': 'person', 'dalmatian': 'dog',
    'chick': 'chicken', 'gondola': 'boat', 'ewe': 'sheep', 'sailor': 'person', 'fighter': 'airplane', 'receiver': 'person',
    'sweeper': 'person', 'settee': 'couch', 'caster': 'person', 'mansion': 'building', 'pecker': 'bird',
    'emperor': 'person', 'smoker': 'person', 'medic': 'person', 'frank': 'hotdog', 'canary': 'bird', 'chestnut': 'nut',
    'lounger': 'chair', 'brat': 'hotdog', 'snoot': 'nose', 'cardigan': 'sweater', 'tangerine': 'mandarin',
    'wrecker': 'truck', 'setter': 'dog', 'sharpie': 'pen', 'jumper': ['person', 'clothing'], 'digger': ['person', 'truck'],
    'prey': 'animal', 'excavator': ['person', 'truck'], 'watchdog': 'dog', 'barker': 'person', 'sphinx': 'statue',
    'brownstone': 'building', 'pussycat': 'cat', 'romper': 'clothing', 'warbler': 'bird', 'schooner': ['boat', 'glass'],
    'trawler': 'boat', 'hatchback': 'car', 'whaler': 'boat', 'jigger': 'glass', 'cock': 'chicken', 'mallet': 'hammer',
    'clipper': 'scissors', 'angler': 'person'
}

nlp = stanza.Pipeline('en', tokenize_no_ssplit=True)
inflect_engine = inflect.engine()
model = AutoModelForMaskedLM.from_pretrained('bert-large-uncased')
device = torch.device('cuda')
model = model.to(device)
model = model.eval()
tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')
mask_str = '[MASK]'

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
    classes = []
    for lemma in synset.lemmas():
        word = lemma.name().lower()
        if word in word_classes:
            return [word]
    classes = []
    hypernyms = synset.hypernyms()
    for hypernym in hypernyms:
        classes += find_synset_classes(hypernym)
    return classes

def is_phrase_hypernym_of_synset(synset, phrase):
    for lemma in synset.lemmas():
        word = lemma.name().lower()
        if word == phrase:
            return True
    hypernyms = synset.hypernyms()
    for hypernym in hypernyms:
        if is_phrase_hypernym_of_synset(hypernym, phrase):
            return True
    return False

def is_phrase_hypernym_of_phrase(phrase1, phrase2):
    synsets = wn.synsets(phrase1)
    for synset in synsets:
        if is_phrase_hypernym_of_synset(synset, phrase2.lower()):
            return True
    return False

def find_phrase_class(phrase):
    phrase = phrase.lower()

    if inflect_engine.singular_noun(phrase) != False:
        phrase = inflect_engine.singular_noun(phrase)

    if phrase in known_mappings:
        phrase_class = known_mappings[phrase]
    elif phrase in word_classes:
        phrase_class = phrase
    elif phrase in non_word_classes:
        return None
    else:
        synsets = wn.synsets(phrase)
        synsets = [synset for synset in synsets if synset.pos() == 'n']
        classes = []
        all_synsets_count = sum([get_synset_count(x) for x in synsets])
        for synset in synsets:
            if all_synsets_count == 0 or get_synset_count(synset)/all_synsets_count > 0.2:
                classes += find_synset_classes(synset)
        classes = list(set(classes))
        if len(classes) == 0:
            phrase_class = None
        else:
            # First, reduce classes to hyponyms only
            to_remove = {}
            for i in range(len(classes)):
                for j in range(i+1, len(classes)):
                    if is_hyponym_of(classes[i], classes[j]):
                        to_remove[j] = True
                    elif is_hyponym_of(classes[j], classes[i]):
                        to_remove[i] = True
            classes = [classes[i] for i in range(len(classes)) if i not in to_remove]

            # If you have a word that can be refered to both as a fruit and as plant (e.g., 'raspberry') choose a fruit
            strong_classes = ['fruit', 'vegetable']
            def is_hyponym_of_strong_class(phrase):
                for strong_class in strong_classes:
                    if is_hyponym_of(phrase, strong_class):
                        return True
                return False
            
            if len(classes) == 2 and is_hyponym_of_strong_class(classes[0]) and is_hyponym_of(classes[1], 'plant'):
                classes = [classes[0]]
            if len(classes) == 2 and is_hyponym_of_strong_class(classes[1]) and is_hyponym_of(classes[0], 'plant'):
                classes = [classes[1]]

            # If we got 2 classes, one of which is a hypernym of the other, we'll take the lower one
            if len(classes) == 2 and is_hyponym_of(classes[0], classes[1]):
                classes = [classes[0]]
            elif len(classes) == 2 and is_hyponym_of(classes[1], classes[0]):
                classes = [classes[1]]

            if len(classes) > 1:
                should_be_handled_list = ['rocker', 'tumbler', 'anemone', 'selector', 'rotisserie', 'bowler', 'digger', 'excavator', 'hydroplane', 'schooner', 'amphibian']
                if phrase in should_be_handled_list:
                    return classes

            # Else, we can't except more than one class
            assert len(classes) == 1, f'Phrase "{phrase}" has multiple classes'
            phrase_class = classes[0]

    return phrase_class

def is_noun(token_list, ind):
    if token_list[ind][0]['upos'] == 'NOUN':
        return True
    
    # "remote" edge cases: in many cases, when people say "remote" they mean "remote controller", i.e., a noun. But the
    #  parser treats it as an adjective. To identify these cases, we'll find "remote" with non-noun heads
    head_ind = token_list[ind][0]['head'] - 1
    if token_list[ind][0]['text'].lower() == 'remote' and token_list[head_ind][0]['upos'] != 'NOUN':
        return True
    
    return False

def is_sequence_punctuation(token_list, ind):
    if token_list[ind][0]['upos'] != 'PUNCT':
        return False
    
    return token_list[ind][0]['text'] == '-'
    
def extract_noun_spans(token_list):
    noun_spans = []

    # First find sequences of nouns
    noun_sequences = []
    in_sequence = False
    for i in range(len(token_list)):
        if is_noun(token_list, i) and (not in_sequence):
            sequence_start = i
            in_sequence = True
        if (not is_noun(token_list, i)) and in_sequence and (not is_sequence_punctuation(token_list, i)):
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

def preprocess(token_list):
    # Just solving some known issues
    
    # Replace phrases to make the parser's job easier
    replace_dict = [
        # 1. Every time we have 'remote control' in a sentence, 'remote' is an adjective so the identified noun span is
        # 'control', which isn't what we want. So we'll change it to 'remote'
        (['remote', 'control'], 'remote'),
        # 2. "hot dog": hot is considered an adjective, and the only identified noun is "dog"
        (['hot', 'dog'], 'hotdog'),
        (['hot', 'dogs'], 'hotdogs'),
        # 2. "olive green": olive is considered a noun
        (['olive', 'green'], 'green')
    ]

    tokens = [x[0]['text'].lower() for x in token_list]
    inds_in_orig_strs = [0]*len(replace_dict)
    to_replace = []
    for i in range(len(tokens)):
        token = tokens[i]
        for j in range(len(replace_dict)):
            if token == replace_dict[j][0][inds_in_orig_strs[j]]:
                inds_in_orig_strs[j] += 1
                if inds_in_orig_strs[j] == len(replace_dict[j][0]):
                    to_replace.append((i - len(replace_dict[j][0]) + 1, i + 1, replace_dict[j][1]))
                    inds_in_orig_strs[j] = 0
            else:
                inds_in_orig_strs[j] = 0

    if len(to_replace) > 0:
        for start_ind, end_ind, new_str in to_replace:
            tokens[start_ind] = new_str
            tokens[start_ind+1:end_ind] = ['[BLANK]']*(end_ind-(start_ind+1))
        tokens = [x for x in tokens if x != '[BLANK]']
        preprocessed_sentence = ' '.join(tokens)
        doc = nlp(preprocessed_sentence)
        token_lists = [[x.to_dict() for x in y.tokens] for y in doc.sentences]
        token_list = token_lists[0]

    return token_list

def get_probs_from_lm(text, returned_vals):
    input = tokenizer(text, return_tensors='pt').to(device)
    mask_id = tokenizer.vocab[mask_str]
    mask_ind = [i for i in range(input.input_ids.shape[1]) if input.input_ids[0, i] == mask_id][0]
    output = model(**input)
    mask_logits = output.logits[0, mask_ind, :]
    if returned_vals == 'logits':
        return mask_logits
    elif returned_vals == 'probs':
        mask_probs = nn.functional.softmax(mask_logits, dim=0)
        return mask_probs
    else:
        assert False

def is_an_word(word):
    inflected = inflect_engine.a(word)
    return inflected.startswith('an')

def choose_class(token_list, start_ind, end_ind, class_list, selection_method='probs'):
    before = [x[0]['text'].lower() for x in token_list[:start_ind]]
    after = [x[0]['text'].lower() for x in token_list[end_ind:]]
    
    # To prevent unwanted bias, check if we need to consider a/an
    if len(before) > 0 and before[-1] in ['a', 'an']:
        a_classes = []
        an_classes = []
        for cur_class in class_list:
            if is_an_word(cur_class):
                an_classes.append(cur_class)
            else:
                a_classes.append(cur_class)
        a_text = ' '.join(before[:-1] + ['a', mask_str] + after)
        a_probs = get_probs_from_lm(a_text, selection_method)
        an_text = ' '.join(before[:-1] + ['an', mask_str] + after)
        an_probs = get_probs_from_lm(an_text, selection_method)
        prob_class_list = [(a_probs, a_classes), (an_probs, an_classes)]
    else:
        text = ' '.join(before + [mask_str] + after)
        probs = get_probs_from_lm(text, selection_method)
        prob_class_list = [(probs, class_list)]
    max_class_prob = (-1)*math.inf
    class_with_max_prob = None
    for probs, classes in prob_class_list:
        for cur_class in classes:
            if cur_class not in tokenizer.vocab:
                # For now, don't handle
                continue
            class_id = tokenizer.vocab[cur_class]
            class_prob = probs[class_id]
            if class_prob > max_class_prob:
                max_class_prob = class_prob
                class_with_max_prob = cur_class
    return class_with_max_prob

def find_classes(caption):
    doc = nlp(caption)
    token_lists = [[x.to_dict() for x in y.tokens] for y in doc.sentences]
    if len(token_lists) > 1:
        return None
    token_list = token_lists[0]
    token_list = preprocess(token_list)

    noun_spans = extract_noun_spans(token_list)
    classes = []

    for start_ind, end_ind, highest_ancestor_ind in noun_spans:
        phrase = ' '.join([token_list[i][0]['text'] for i in range(start_ind, end_ind)]).lower()
        phrase_class = find_phrase_class(phrase)

        # Check only the highest ancestor in the noun span
        if phrase_class is None:
            phrase = token_list[highest_ancestor_ind][0]['text']
            phrase_class = find_phrase_class(phrase)

        # 2. We have a problem when there's a sport named the same as its ball (baseball, basketball etc.).
        # The more common synset is the game, and when someone talks about the ball the algorithm always thinks it's the game.
        # We'll try identifying these cases by checking if it's a single noun and there's an identifier before it
        if phrase_class is None \
            and end_ind - start_ind == 1 \
            and start_ind > 0 \
            and token_list[start_ind-1][0]['upos'] == 'DET' \
            and token_list[start_ind][0]['text'].endswith('ball'):
            phrase_class = 'ball'

        if type(phrase_class) is list:
            phrase_class = choose_class(token_list, start_ind, end_ind, phrase_class)

        classes.append((start_ind, end_ind, phrase_class))
    
    return classes
