import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import sys
import json
sys.path.append('.')
from utils.general_utils import all_synsets, child2parent, get_synset_to_image_prob, parent2children
import scipy.stats as stats
from streamlit_app.app_utils import plot_clickable_images
from collections import OrderedDict

debug = False
def debug_print(my_str):
    if debug:
        print(my_str)

# Initalize
state = st.session_state
lang_names_and_codes = ['Arabic (ar)', 'Chinese-Simplified (zh)', 'Croatian (hr)', 'Czech (cs)', 'Danish (da)', 'Dutch (nl)', 'English (en)', 'Filipino (fil)', 'Finnish (fi)', 'French (fr)', 'German (de)', 'Greek (el)', 'Hebrew (he)', 'Hindi (hi)', 'Hungarian (hu)', 'Indonesian (id)', 'Italian (it)', 'Japanese (ja)', 'Korean (ko)', 'Norwegian (no)', 'Persian (fa)', 'Polish (pl)', 'Portuguese (pt)', 'Romanian (ro)', 'Russian (ru)', 'Spanish (es)', 'Swedish (sv)', 'Thai (th)', 'Turkish (tr)', 'Ukrainian (uk)', 'Vietnamese (vi)']
lang_names = [x.split(' (')[0] for x in lang_names_and_codes]
lang_codes = [x.split('(')[1].split(')')[0] for x in lang_names_and_codes]
lang_name2code = {lang_names[i]: lang_codes[i] for i in range(len(lang_names))}
lang_code2name = {lang_codes[i]: lang_names[i] for i in range(len(lang_names))}

root_synsets = set([x for x in all_synsets if x not in child2parent])

# Each new time someone enters the app, the state is re initialized. So we need to reload the worksheet and data
if 'cur_page' not in state:
    state.cur_page = 0
    state.languages = None
    state.concept = None

def to_language_selection_page():
    debug_print('In to_language_selection_page', flush=True)
    move_to(1)

def to_root_concept_selection_page():
    if 'language_selection_box0' in state:
        state.languages = [state[f'language_selection_box{i}'] for i in range(state.language_num)]
    debug_print('In to_root_concept_selection_page', flush=True)
    move_to(2)

def to_sub_concept_selection_page():
    debug_print('In to_sub_concept_selection_page', flush=True)
    state.root_concept = state.root_concept_selection_box
    move_to(3)

def to_language_by_concept_analysis_page():
    if 'language_selection_box0' in state:
        state.languages = [state[f'language_selection_box{i}'] for i in range(state.language_num)]
    debug_print('In to_language_by_concept_analysis_page', flush=True)
    if state.concept is not None:
        move_to(4)

def to_image_page():
    debug_print('In to_image_page', flush=True)
    move_to(5)

def to_two_languages_selection_page():
    debug_print('In to_two_languages_selection_page', flush=True)
    move_to(6)

def to_concept_across_all_languages_page():
    debug_print('In to_concept_across_all_languages_page', flush=True)
    move_to(7)

def move_to(page_ind):
    debug_print(f'In move_to with page_ind {page_ind}', flush=True)
    debug_print(f'cur_page before: {state.cur_page}', flush=True)
    state.cur_page = page_ind
    debug_print(f'cur_page after: {state.cur_page}', flush=True)
    if state.cur_page == 5:
        st.rerun()

def menu_page():
    st.header('Cross-Lingual and Cross-Cultural Variation in Image Descriptions')

    st.subheader('Overview')
    st.markdown('This tool allows you to inspect variations in image descriptions across languages, locations and concepts. For more information read our paper.')
    st.markdown('If you use this tool please cite:')

    st.subheader('Preliminaries')
    st.markdown('We use the CrossModal3600 dataset and extract the concepts mentioned in each caption, for each image in each language.')
    st.markdown('The concepts are WordNet synsets.')
    st.markdown('You can use this tool to analyze the data and examine specific data points.')

    st.button('Analyze by language', key='by_language_button', on_click=to_language_selection_page)
    st.button('Analyze by concept', key='by_concept_button', on_click=to_root_concept_selection_page)
    st.button('Compare languages', key='compare_languages_button', on_click=to_two_languages_selection_page)

def language_selection_page(language_num=1):
    st.header('Language selection')
    state.language_num = language_num
    for i in range(language_num):
        if language_num == 1:
            button_text = 'Please select language:'
        else:
            button_text = f'Please select language {i+1}:'
        st.selectbox(label=button_text, key=f'language_selection_box{i}', options=lang_names)
    if state.concept is None:
        st.button('Continue', key='language_page_continue_button', on_click=to_root_concept_selection_page)
    else:
        st.button('Continue', key='language_page_continue_button', on_click=to_language_by_concept_analysis_page)
        st.markdown('OR')
        st.button(f'Press here to analyze {state.concept} across all languages', key='analyze_concept_across_all_languages_button', on_click=to_concept_across_all_languages_page)

def root_concept_selection_page():
    if state.languages is not None:
        st.header(f'Analyze by languages: {",".join(state.languages)}')
    st.selectbox(label='Please select root concept:', key='root_concept_selection_box', options=root_synsets)
    st.button('Continue', key='language_analysis_page_continue_button', on_click=to_sub_concept_selection_page)

def sub_concept_selection_page():
    language_str = '' if state.languages is None else f', languages: {",".join(state.languages)}'
    st.header(f'Root concept: {state.root_concept}{language_str}')
    st.markdown('Please select a concept from the subtree:')
    nodes = []
    edges = []
    root_concept = state.root_concept
    queue = [root_concept]
    while len(queue) > 0:
        cur_concept = queue[0]
        nodes.append(Node(id=cur_concept, label=cur_concept, size=25, shape='rectangle'))
        queue = queue[1:]
        children = parent2children[cur_concept]
        for child in children:
            queue.append(child)
            edges.append(Edge(source=cur_concept, target=child))

    config = Config(width=600, height=400, directed=True, physics=True, hierarchical=True)

    state.concept = agraph(nodes=nodes, 
                        edges=edges, 
                        config=config)
    
    if state.languages is None:
        st.button('Continue', key='sub_concept_selection_page_continue_button', on_click=to_language_selection_page)
    else:
        st.button('Continue', key='sub_concept_selection_page_continue_button', on_click=to_language_by_concept_analysis_page)

def by_language_concept_analysis_page():
    st.header('Analysis')
    st.markdown(f'Analyzing: {state.concept} in captions in {",".join(state.languages)}')
    if len(state.languages) == 1:
        by_single_language_concept_analysis_page()
    else:
        by_two_languages_concept_analysis_page()

def by_single_language_concept_analysis_page():
    lang_code = state.languages[0].split('(')[1].split(')')[0]
    synset_to_image_prob, _, _ = get_synset_to_image_prob(f'xm3600_{lang_code}')
    image_num = len([x for x in synset_to_image_prob[state.concept].values() if x > 0])
    st.subheader('Statistics')
    st.markdown(f'Out of the 3600 images in the datasets, {image_num} include captions mentioning {state.concept}')
    if image_num > 0:
        prob_sum = sum(synset_to_image_prob[state.concept].values())
        prob_mean = prob_sum/image_num
        st.markdown(f'Each image is annotated by multiple annotators- some may mention the concept, while other doesn\'t. In these {image_num} images, the average fraction of annotators mentioning {state.concept} is {prob_mean}')

        st.subheader('Images')
        iids = list(synset_to_image_prob[state.concept].keys())
        clicked = plot_clickable_images(iids)
        if clicked > -1:
            state.iid = iids[clicked]
            to_image_page()

def by_two_languages_concept_analysis_page():
    lang_code1 = state.languages[0].split('(')[1].split(')')[0]
    synset_to_image_prob1, _, _ = get_synset_to_image_prob(f'xm3600_{lang_code1}')
    image_num1 = len([x for x in synset_to_image_prob1[state.concept].values() if x > 0])
    lang_code2 = state.languages[1].split('(')[1].split(')')[0]
    synset_to_image_prob2, _, _ = get_synset_to_image_prob(f'xm3600_{lang_code2}')
    image_num2 = len([x for x in synset_to_image_prob2[state.concept].values() if x > 0])
    st.subheader('Statistics')
    st.markdown(f'Out of the 3600 images in the datasets, {image_num1} include captions in {state.languages[0]} and {image_num2} include captions in {state.languages[1]} mentioning {state.concept}')
    if image_num1 > 0:
        prob_sum1 = sum(synset_to_image_prob1[state.concept].values())
        prob_mean1 = prob_sum1/image_num1
    else:
        prob_mean1 = 0
    if image_num2 > 0:
        prob_sum2 = sum(synset_to_image_prob2[state.concept].values())
        prob_mean2 = prob_sum2/image_num2
    else:
        prob_mean2 = 0
    if prob_mean1 + prob_mean2 > 0:
        st.markdown(f'Each image is annotated by multiple annotators- some may mention the concept, while other doesn\'t.')
    if prob_mean1:
        st.markdown(f'For {state.languages[0]}, in these {image_num1} images, the average fraction of annotators mentioning {state.concept} is {prob_mean1}')
    if prob_mean2:
        st.markdown(f'For {state.languages[1]}, in these {image_num2} images, the average fraction of annotators mentioning {state.concept} is {prob_mean2}')

    with open(f'datasets/xm3600_{lang_code1}.json', 'r') as fp:
        data1 = json.load(fp)
    iids = list(set([x['image_id'] for x in data1]))
    pval1 = stats.wilcoxon([synset_to_image_prob1[state.concept][x] if x in synset_to_image_prob1[state.concept] else 0 for x in iids], [synset_to_image_prob2[state.concept][x] if x in synset_to_image_prob2[state.concept] else 0 for x in iids], alternative='greater', zero_method='zsplit').pvalue
    pval2 = stats.wilcoxon([synset_to_image_prob1[state.concept][x] if x in synset_to_image_prob1[state.concept] else 0 for x in iids], [synset_to_image_prob2[state.concept][x] if x in synset_to_image_prob2[state.concept] else 0 for x in iids], alternative='less', zero_method='zsplit').pvalue
    if pval1 < pval2:
        first_lang = state.languages[0]
        not_str = 'not ' if pval1 >= 0.05 else ''
        pval = pval1
    else:
        first_lang = state.languages[1]
        not_str = 'not ' if pval2 >= 0.05 else ''
        pval = pval2
    st.markdown(f'**Significance test**: When running the Wilcoxon signed-ranked test on the average fractions of annotators mentioning {state.concept} for each image, {first_lang} was found more likely to mention it ({not_str}significant, pvalue={pval})')

    iids1 = [x[0] for x in synset_to_image_prob1[state.concept].items() if x[1] == 1 and (x[0] not in synset_to_image_prob2[state.concept] or synset_to_image_prob2[state.concept][x[0]] == 0)]
    clicked1 = -1
    if len(iids1) > 0:
        st.subheader(f'Images where all annotators in {state.languages[0]} mentioned {state.concept} while none of the annotators in {state.languages[1]} did so')
        clicked1 = plot_clickable_images(iids1)

    iids2 = [x[0] for x in synset_to_image_prob2[state.concept].items() if x[1] == 1 and (x[0] not in synset_to_image_prob1[state.concept] or synset_to_image_prob1[state.concept][x[0]] == 0)]
    clicked2 = -1
    if len(iids2) > 0:
        st.subheader(f'Images where all annotators in {state.languages[1]} mentioned {state.concept} while none of the annotators in {state.languages[0]} did so')
        clicked2 = plot_clickable_images(iids2)

    if clicked1 > -1:
        state.iid = iids1[clicked1]
        to_image_page()
    if clicked2 > -1:
        state.iid = iids2[clicked2]
        to_image_page()

def image_page():
    image_file_path = f'/mnt/c/Users/uribe/PycharmProjects/datasets/crossmodal3600/images/{hex(state.iid)[2:].zfill(16)}.jpg'
    st.image(image_file_path)

    st.subheader('Captions')
    for lang in state.languages:
        lang_code = lang.split('(')[1].split(')')[0]
        with open(f'datasets/xm3600_{lang_code}.json', 'r') as fp:
            data = json.load(fp)
        samples = [x for x in data if x['image_id'] == state.iid]
        for sample in samples:
            with st.container(border=True):
                if 'orig' in sample:
                    st.markdown(f'Original caption: {sample["orig"]}')
                    st.markdown(f'English translated caption: {sample["caption"]}')
                else:
                    st.markdown(f'Caption: {sample["caption"]}')
                st.markdown(f'Identified concepts: {[("" + x[2] + "", x[3]) for x in sample["synsets"]]}')

    st.button('Back', key='image_back_button', on_click=to_language_by_concept_analysis_page)

def concept_analysis_across_all_languages_page():
    st.header(f'Mean {state.concept} saliency across languages')
    res = OrderedDict()
    for lang in lang_names:
        debug_print(lang)
        lang_code = lang_name2code[lang]
        synset_to_image_prob, _, _ = get_synset_to_image_prob(f'xm3600_{lang_code}')
        # res[lang_code2name[lang]] = sum(synset_to_image_prob[state.concept].values())/3600
        res[lang] = sum(synset_to_image_prob[state.concept].values())/3600
        # df[i] = [lang, sum(synset_to_image_prob[state.concept].values())/3600]
    st.bar_chart(res, y_label='Mean saliency')
    st.markdown(f'This plot presents the mean saliency of {state.concept} in each language.')
    st.markdown(f'The saliency of {state.concept} in a specific image in language $L$ is the fraction of annotators of the image in $L$ mentioning {state.concept}.')
    st.markdown(f'The overall saliency in $L$ is computed by averaging across all 3600 images in the CrossModal3600 dataset.')

debug_print(f'cur_page: {state.cur_page}', flush=True)
if state.cur_page == 0:
    menu_page()
elif state.cur_page == 1:
    language_selection_page()
elif state.cur_page == 2:
    root_concept_selection_page()
elif state.cur_page == 3:
    sub_concept_selection_page()
elif state.cur_page == 4:
    by_language_concept_analysis_page()
elif state.cur_page == 5:
    image_page()
elif state.cur_page == 6:
    language_selection_page(2)
elif state.cur_page == 7:
    concept_analysis_across_all_languages_page()
