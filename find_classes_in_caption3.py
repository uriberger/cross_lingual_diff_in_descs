import stanza
from nltk.corpus import wordnet as wn
import inflect
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from torch import nn
import math
import clip
from PIL import Image

class_phrases = [
    'person', 'vehicle', 'furniture', 'animal', 'food', 'bag', 'clothing', 'tableware', 'plant', 'electronic_equipment',
    'home_appliance', 'toy', 'building', 'mountain', 'kitchen_utensil', 'sky', 'celestial_body', 'body_part',
    'body_of_water', 'hand_tool', 'musical_instrument', 'writing_implement', 'jewelry', 'weapon', 'timepiece'
    # Riding device
    'riding_device', 'ski', 'surfboard', 'snowboard', 'skateboard', 'rollerblade',
    # Things I added manually
    'factory'
    # Things that are not under any class and I don't know what to do with them
    #'ball', 'computer', 'book', 'door', 'window', 'bridge'
    ]

parent_to_children3 = {
    'riding_device': ['ski', 'surfboard', 'snowboard', 'skateboard', 'rollerblade'],
    'tableware': ['plate'],
    'pepper': ['chili_pepper'],
    'building': ['factory']
}

child_to_parent3 = {}
for parent, children in parent_to_children3.items():
    for child in children:
        child_to_parent3[child] = parent

def is_hyponym_of(class1, class2):
    if class1 == class2:
        return True
    while class1 in child_to_parent3:
        return is_hyponym_of(child_to_parent3[class1], class2)
    return False

non_class_phrases = [
    'sport', 'amazon', 'quarry', 'aa', 'cob', 'chat', 'maroon', 'white', 'header', 'gravel', 'black', 'bleachers',
    'middle', 'lot', 'lots', 'gear', 'rear', 'bottom', 'nationality', 'overlay', 'city_center', 'center', 'recording',
    'lid', 'region', 'meal', 'pair', 'upside', 'front', 'left', 'exterior', 'an', 'elderly', 'young', 'small_white',
    'small', 'blue', 'skate', 'third', 'aged', 'styrofoam', 'adult', 'dome', 'stadium', 'granite', 'machine', 'string'
]

# Inflect don't handle some strings well, ignore these
non_inflect_strs3 = [
    'dress'
]

sister_term_mappings = {
    'people': 'person'
}

hypernym_mappings = {
    # 'snowboarder': 'person', 'surfer': 'person', 'scooter': 'motorcycle', 'hay': 'plant', 'van': 'car', 'walnut': 'nut',
    # 'peanut': 'nut', 'diner': 'restaurant', 'guy': 'man', 'toy_car': 'toy', 'plantain': 'banana', 'racer': 'person',
    # 'pet': 'animal', 'president': 'person', 'guide': 'person', 'climber': 'person', 'commuter': 'person',
    # 'dalmatian': 'dog', 'gondola': 'boat', 'ewe': 'sheep', 'sailor': 'person', 'fighter': 'airplane', 'receiver': 'person',
    # 'sweeper': 'person', 'settee': 'couch', 'caster': 'person', 'mansion': 'building', 'pecker': 'bird',
    # 'emperor': 'person', 'smoker': 'person', 'medic': 'person', 'canary': 'bird', 'chestnut': 'nut', 'lounger': 'chair',
    # 'cardigan': 'sweater', 'wrecker': 'truck', 'setter': 'dog', 'jumper': ['person', 'clothing'],
    # 'digger': ['person', 'truck'], 'prey': 'animal', 'excavator': ['person', 'truck'], 'watchdog': 'dog',
    # 'barker': 'person', 'sphinx': 'statue', 'brownstone': 'building', 'romper': 'clothing', 'warbler': 'bird',
    # 'schooner': ['boat', 'glass'], 'trawler': 'boat', 'hatchback': 'car', 'whaler': 'boat', 'jigger': 'cup',
    # 'angler': 'person', 'weaver': 'person', 'predator': 'animal', 'arab': 'ethnic_group', 'asian': 'ethnic_group',
    # 'african': 'ethnic_group', 'hulk': 'person', 'outfit': 'clothing', 'jean': 'pant', 'back': ['body_part', None],
    # 'shorts': 'clothing', 'tower': 'building', 'factory': 'building', 'fortress': 'building', 'fort': 'building',
    # 'subway': 'train', 'lavender': 'flower', 'dish': 'tableware', 'butt': 'body_part', 'python': 'snake',
    # 'saucer': 'tableware',
}

word_to_replace_str3 = {
    # 'back': {'body_part': 'hand', None: 'rear'}, 'glasses': {'cup': 'cups', 'eyeglasses': 'sunglasses'},
    # 'dish': {'dish': 'dish', 'tableware': 'plate'}
}

nlp = stanza.Pipeline('en', tokenize_no_ssplit=True)
inflect_engine = inflect.engine()
bert_model = AutoModelForMaskedLM.from_pretrained('bert-large-uncased')
device = torch.device('cuda')
bert_model = bert_model.to(device)
bert_model = bert_model.eval()
tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')
mask_str = '[MASK]'
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

def get_synset_count(synset):
    count = 0
    for lemma in synset.lemmas():
        count += lemma.count()
    return count

def find_synset_classes3(synset):
    for lemma in synset.lemmas():
        word = lemma.name().lower()
        if word in class_phrases:
            return [[word, 0]]
    classes = []
    hypernyms = synset.hypernyms()
    for hypernym in hypernyms:
        cur_classes = find_synset_classes3(hypernym)
        for i in range(len(cur_classes)):
            cur_classes[i][1] += 1
        classes += cur_classes
    return classes

def find_phrase_classes3(phrase):
    phrase = phrase.lower()

    # First, preprocess: if in plural, convert to singular
    singular_phrase_classes = None
    if phrase not in non_inflect_strs3 and inflect_engine.singular_noun(phrase) != False:
        singular_phrase = inflect_engine.singular_noun(phrase)
        singular_phrase_classes = find_preprocessed_phrase_classes3(singular_phrase)

    if singular_phrase_classes is not None and len(singular_phrase_classes) > 0 and len([x for x in singular_phrase_classes if x[0] is not None]) > 0:
        return singular_phrase_classes
    else:
        return find_preprocessed_phrase_classes3(phrase)

def search_in_wordnet(phrase):
    synsets = wn.synsets(phrase)
    synsets = [synset for synset in synsets if synset.pos() == 'n']
    classes = []
    all_synsets_count = sum([get_synset_count(x) for x in synsets])
    for synset in synsets:
        if all_synsets_count == 0 or get_synset_count(synset)/all_synsets_count >= 0.2:
            classes += find_synset_classes3(synset)

    class_to_lowest_num = {}
    for cur_class, num in classes:
        if cur_class not in class_to_lowest_num or num < class_to_lowest_num[cur_class]:
            class_to_lowest_num[cur_class] = num

    classes = list(class_to_lowest_num.items())
    if len(classes) == 0:
        return [(None, 0)]
    else:
        # First, reduce classes to hyponyms only
        to_remove = {}
        for i in range(len(classes)):
            for j in range(i+1, len(classes)):
                if is_hyponym_of(classes[i][0], classes[j][0]):
                    to_remove[j] = True
                elif is_hyponym_of(classes[j][0], classes[i][0]):
                    to_remove[i] = True
        classes = [classes[i] for i in range(len(classes)) if i not in to_remove]

        # If you have a word that can be refered to both as a fruit and as plant (e.g., 'raspberry') choose a fruit
        strong_classes = ['fruit', 'vegetable']
        def is_hyponym_of_strong_class(phrase):
            for strong_class in strong_classes:
                if is_hyponym_of(phrase, strong_class):
                    return True
            return False
        
        if len(classes) == 2 and is_hyponym_of_strong_class(classes[0][0]) and is_hyponym_of(classes[1][0], 'plant'):
            classes = [classes[0]]
        if len(classes) == 2 and is_hyponym_of_strong_class(classes[1][0]) and is_hyponym_of(classes[0][0], 'plant'):
            classes = [classes[1]]

        # If we got 2 classes, one of which is a hypernym of the other, we'll take the lower one
        if len(classes) == 2 and is_hyponym_of(classes[0][0], classes[1][0]):
            classes = [classes[0]]
        elif len(classes) == 2 and is_hyponym_of(classes[1][0], classes[0][0]):
            classes = [classes[1]]

    return classes

def find_preprocessed_phrase_classes3(phrase):
    phrase = phrase.replace(' ', '_')

    ''' Phrase class may be found in:
    1. Known mappings from phrases to classes.
    2. Exact match: the phrase is in the list of class phrases
    3. Exact mismatch: the phrase is in the list of non-class phrases
    4. WordNet: Search in the wordnet onthology
    '''

    phrase_mappings = []
    if phrase in sister_term_mappings:
        sister_term_mapping = sister_term_mappings[phrase]
        if type(sister_term_mapping) == list:
            phrase_mappings += [(x, 0) for x in sister_term_mapping]
        else:
            phrase_mappings.append((sister_term_mapping, 0))
    # if phrase in hypernym_mappings:
    #     hypernym_mapping = hypernym_mappings[phrase]
    #     if type(hypernym_mapping) == list:
    #         phrase_mappings += [(x, False) for x in hypernym_mapping]
    #     else:
    #         phrase_mappings.append((hypernym_mapping, False))

    if len(phrase_mappings) > 0:
        # 1. Known mappings
        return phrase_mappings
    elif phrase in class_phrases:
        # 2. Exact match
        return [(phrase, 0)]
    elif phrase in non_class_phrases:
        # Exact mismatch
        return [(None, 0)]
    else:
        # Wordnet
        return search_in_wordnet(phrase)

def preprocess(token_list):
    # Just solving some known issues
    
    # Replace phrases to make the parser's job easier
    replace_dict = [
        # 1. "olive green": olive is considered a noun
        (['olive', 'green'], 'green'),
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
    input = tokenizer(text, return_tensors='pt', truncation='longest_first').to(device)
    mask_id = tokenizer.vocab[mask_str]
    mask_ind = [i for i in range(input.input_ids.shape[1]) if input.input_ids[0, i] == mask_id][0]
    output = bert_model(**input)
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

def choose_class_with_lm(token_list, start_ind, end_ind, class_list, selection_method='probs'):
    class_to_dist_from_match = {x[0]: x[1] for x in class_list}
    only_class_list = [x[0] for x in class_list]

    before = [x[0]['text'].lower() for x in token_list[:start_ind]]
    after = [x[0]['text'].lower() for x in token_list[end_ind:]]

    orig_word = '_'.join([x[0]['text'] for x in token_list[start_ind:end_ind]])
    if orig_word in word_to_replace_str3:
        class_to_repr_word = word_to_replace_str3[orig_word]
    else:
        class_to_repr_word = {cur_class: cur_class for cur_class in only_class_list}
    
    # To prevent unwanted bias, check if we need to consider a/an
    if len(before) > 0 and before[-1] in ['a', 'an']:
        a_classes = []
        an_classes = []
        for cur_class in only_class_list:
            if is_an_word(class_to_repr_word[cur_class]):
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
        prob_class_list = [(probs, only_class_list)]

    max_class_prob = (-1)*math.inf
    class_with_max_prob = None
    for probs, classes in prob_class_list:
        for cur_class in classes:
            repr_word = class_to_repr_word[cur_class]
            if repr_word not in tokenizer.vocab:
                # For now, don't handle
                continue
            class_id = tokenizer.vocab[repr_word]
            class_prob = probs[class_id]
            if class_prob > max_class_prob:
                max_class_prob = class_prob
                class_with_max_prob = cur_class

    if class_with_max_prob is None:
        dist_from_match = 0
    else:
        dist_from_match = class_to_dist_from_match[class_with_max_prob]
    return class_with_max_prob, dist_from_match

def choose_class_with_clip(token_list, start_ind, end_ind, class_list, image_path):
    before = [x[0]['text'].lower() for x in token_list[:start_ind]]
    after = [x[0]['text'].lower() for x in token_list[end_ind:]]

    class_text_list = []
    # To prevent unwanted bias, check if we need to consider a/an
    if len(before) > 0 and before[-1] in ['a', 'an']:
        for cur_class in class_list:
            if is_an_word(cur_class):
                det_str = 'an'
            else:
                det_str = 'a'
            class_text_list.append((cur_class, ' '.join(before[:-1] + [det_str, cur_class] + after)))
    else:
        class_text_list = [(cur_class, ' '.join(before + [cur_class] + after)) for cur_class in class_list]

    image = clip_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize([x[1] for x in class_text_list]).to(device)

    with torch.no_grad():
        logits_per_image, _ = clip_model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    return class_list[probs.argmax()]

def is_subtree_first(token_list, ind):
    adjusted_ind = ind + 1
    subtree_first = True
    for cur_ind in range(ind):
        # Check if its an ancestor of ind
        inner_ind = cur_ind
        while True:
            if token_list[inner_ind][0]['head'] == adjusted_ind:
                break
            if token_list[inner_ind][0]['head'] > adjusted_ind or token_list[inner_ind][0]['head'] == 0:
                subtree_first = False
                break
            inner_ind = token_list[inner_ind][0]['head'] - 1
        if not subtree_first:
            break
    return subtree_first

def has_determiner(token_list, ind):
    return len([x for x in token_list if x[0]['head'] == ind+1 and x[0]['upos'] == 'DET']) > 0

def ball_handling(token_list, ball_ind):
    # Plural is always the ball, never the game
    if token_list[ball_ind][0]['text'].endswith('balls'):
        return 'ball', token_list[ball_ind][0]['text'] == 'balls'

    # Paintball is not a ball
    if token_list[ball_ind][0]['text'] == 'paintball':
        return None, False
    
    # If it's a single word at the beginning of the sentence or with a determiner before it- it's the ball,
    # otherwise it's the game
    if is_subtree_first(token_list, ball_ind):
        return 'ball', token_list[ball_ind][0]['text'] == 'ball'
    
    if has_determiner(token_list, ball_ind):
        return 'ball', token_list[ball_ind][0]['text'] == 'ball'
    
    return None, False

def top_handling(token_list, start_ind):
    # Need to distinguish top as a preposition from the clothing
    if len([
        x for x in token_list if x[0]['head'] == start_ind+1 and
        x[0]['upos'] == 'DET' and
        x[0]['text'].lower() in ['a', 'an']
        ]) > 0:
        return 'clothing', 2
    
    return None, 0

def couple_handling(token_list, ind):
    # If we have "a couple of..." we don't want it to have a class, if it's "A couple sitting on a bench"
    # we do want. Distinguish by checking if we have no "of" after it
    if ind < (len(token_list) - 1) and token_list[ind+1][0]['text'].lower() == 'of':
        return None, 0
    
    return 'person', 1

def plant_handling(token_list, start_ind, end_ind):
    # If we have a plant, it's the living thing- unless the word "power" is before it
    if end_ind - start_ind == 2 and token_list[start_ind][0]['text'] == 'power':
        return 'factory', 0
    
    return 'plant', 0

def phrase_location_to_class3(token_list, start_ind, end_ind):
    phrase = ' '.join([token_list[i][0]['text'] for i in range(start_ind, end_ind)]).lower()

    # 1. We have a problem when there's a sport named the same as its ball (baseball, basketball etc.).
    # The more common synset is the game, and when someone talks about the ball the algorithm always thinks it's the game.
    # We'll try identifying these cases
    # if end_ind - start_ind == 1 and token_list[start_ind][0]['text'].endswith('ball') or token_list[start_ind][0]['text'].endswith('balls'):
    #     phrase_class, exact_match = ball_handling(token_list, start_ind)

    # 2. "top" is also a problem, as it might be clothing
    if end_ind - start_ind == 1 and token_list[start_ind][0]['text'] == 'top':
        phrase_class, dist_from_match = top_handling(token_list, start_ind)

    # 3. "couple": if we have "a couple of..." we don't want it to have a class, if it's "A couple sitting on a bench"
    # we do want. Distinguish by checking if we have a determiner (or this is the first phrase), and no "of" after it
    elif end_ind - start_ind == 1 and token_list[start_ind][0]['text'] in ['couple', 'couples']:
        phrase_class, dist_from_match = couple_handling(token_list, start_ind)

    # 4. "plant": people almost always mean plants and not factories. We'll always chooce plants except if we see the
    # word "power" before
    elif token_list[end_ind - 1][0]['text'] in ['plant', 'plants']:
        phrase_class, dist_from_match = plant_handling(token_list, start_ind, end_ind)

    else:
        phrase_classes = find_phrase_classes3(phrase)

        if len(phrase_classes) > 1:
            phrase_class, dist_from_match = choose_class_with_lm(token_list, start_ind, end_ind, phrase_classes)
        else:
            phrase_class, dist_from_match = phrase_classes[0]

    return phrase_class, dist_from_match

def is_noun(token_list, ind):
    head_ind = token_list[ind][0]['head'] - 1

    if token_list[ind][0]['upos'] == 'NOUN':
        # VBN edge cases: If we have a noun with a VBN parent (e.g., "flower-covered") the entire phrase is not a noun
        if token_list[head_ind][0]['xpos'] == 'VBN' and token_list[ind][0]['deprel'] == 'compound':
            return False
        
        # "mini" edge case: when used as an adjective parser may call it a noun compound
        if token_list[ind][0]['text'] == 'mini' and token_list[ind][0]['deprel'] == 'compound':
            return False
        
        # uniform edge case: if the word "uniform" follows (e.g., "nurse uniform") this is not a noun
        if ind < len(token_list) - 1 and token_list[ind+1][0]['text'] == 'uniform':
            return False
        
        # glass edge case: if the word "glass" is followed by a noun (e.g., "glass door") this is not a noun
        if token_list[ind][0]['text'] == 'glass' and ind < len(token_list) - 1 and token_list[ind+1][0]['upos'] == 'NOUN':
            return False
        
        # tooth/teeth edge case: if the word "teeth"/"tooth" follows (e.g., "animal teeth") this is not a noun
        if ind < len(token_list) - 1 and token_list[ind+1][0]['text'] in ['teeth', 'tooth']:
            return False
        
        return True
    
    # "remote" edge cases: in many cases, when people say "remote" they mean "remote controller", i.e., a noun. But the
    #  parser treats it as an adjective. To identify these cases, we'll find "remote" with non-noun heads
    if token_list[ind][0]['text'].lower() == 'remote' and token_list[head_ind][0]['upos'] != 'NOUN':
        return True
    
    # "baked goods" edge case: baked is considered adjective, but both should be considered a noun together
    if token_list[ind][0]['text'] == 'baked' and ind < (len(token_list) - 1) and token_list[ind+1][0]['text'] == 'goods':
        return True
    
    # "orange slices" edge case: orange is considered adjective, but both should be considered a noun together
    if token_list[ind][0]['text'] == 'orange' and ind < (len(token_list) - 1) and token_list[ind+1][0]['text'] in ['slice', 'slices']:
        return True
    
    # "german shepherd" edge case: german is considered adjective, but both should be considered a noun together
    if token_list[ind][0]['text'] == 'german' and ind < (len(token_list) - 1) and token_list[ind+1][0]['text'] == 'shepherd':
        return True
    
    return False

def postprocessing(classes):
    # In many cases we have two subsequent nouns referring to the same thing, where the first is a hyponym of the second
    # (e.g., "ferry boat"). In this case we want to reduce the two to one
    final_classes = []
    prev_sample = None
    for sample in classes:
        if prev_sample is not None and \
            prev_sample[0] == prev_sample[1] - 1 and \
            prev_sample[1] == sample[0] and \
            sample[0] == sample[1] - 1 and \
            is_hyponym_of(prev_sample[3], sample[3]):
            final_classes = final_classes[:-1]
            final_classes.append((prev_sample[0], sample[1], prev_sample[2], prev_sample[3], prev_sample[4]))
        else:
            final_classes.append(sample)
        prev_sample = sample

    return final_classes

def find_classes3(caption):
    ''' Count not only noun phrases, but all words. This currently doesn't work well. '''
    caption = caption.lower()
    doc = nlp(caption)
    token_lists = [[x.to_dict() for x in y.tokens] for y in doc.sentences]
    if len(token_lists) > 1:
        return None
    token_list = token_lists[0]
    token_list = preprocess(token_list)

    classes = []

    identified_inds = set()
    # Two word phrases
    i = 0
    while i < len(token_list)-1:
        start_ind = i
        end_ind = i+2
        phrase_class = None
        if is_noun(token_list, i) and is_noun(token_list, i+1):
            phrase_class, dist_from_match = phrase_location_to_class3(token_list, start_ind, end_ind)
        if phrase_class is not None:
            phrase = ' '.join([token_list[i][0]['text'] for i in range(start_ind, end_ind)]).lower()
            classes.append((start_ind, end_ind, phrase, phrase_class, dist_from_match))
            identified_inds.add(start_ind)
            identified_inds.add(start_ind+1)
            i += 2
        else:
            i += 1
    
    # Single word phrases
    for i in range(len(token_list)):
        if i in identified_inds:
            continue
        start_ind = i
        end_ind = i+1
        phrase_class = None
        if is_noun(token_list, i):
            phrase_class, dist_from_match = phrase_location_to_class3(token_list, start_ind, end_ind)
        if phrase_class is not None:
            phrase = ' '.join([token_list[i][0]['text'] for i in range(start_ind, end_ind)]).lower()
            classes.append((start_ind, end_ind, phrase, phrase_class, dist_from_match))

    classes = postprocessing(classes)
    
    return classes
