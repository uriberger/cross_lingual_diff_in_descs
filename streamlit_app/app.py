import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from st_clickable_images import clickable_images
import sys
import json
import base64
sys.path.append('.')
from utils.general_utils import all_synsets, child2parent, get_synset_to_image_prob, parent2children

# Initalize
state = st.session_state

root_synsets = set([x for x in all_synsets if x not in child2parent])

# Each new time someone enters the app, the state is re initialized. So we need to reload the worksheet and data
if 'cur_page' not in state:
    state.cur_page = 0

def to_language_page():
    print('In to_language_page', flush=True)
    move_to(1)

def to_language_analysis_page():
    state.language = state.language_selection_box
    print('In to_language_analysis_page', flush=True)
    move_to(2)

def to_language_by_concept_page():
    print('In to_language_by_concept_page', flush=True)
    state.root_concept = state.root_concept_selection_box
    move_to(3)

def to_language_by_concept_analysis_page():
    print('In to_language_by_concept_analysis_page', flush=True)
    if state.concept is not None:
        move_to(4)

def to_image_page():
    print('In to_image_page', flush=True)
    move_to(5)

def move_to(page_ind):
    print(f'In move_to with page_ind {page_ind}', flush=True)
    print(f'cur_page before: {state.cur_page}', flush=True)
    state.cur_page = page_ind
    print(f'cur_page after: {state.cur_page}', flush=True)
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

    st.button('Analyze by language', key='by_language_button', on_click=to_language_page)

def language_selection_page():
    st.header('Analyze by language')
    st.selectbox(label='Please select language:', key='language_selection_box', options=['Arabic (ar)', 'Chinese-Simplified (zh)', ' Croatian (hr)', ' Czech (cs)', ' Danish (da)', ' Dutch (nl)', ' English (en)', ' Filipino (fil)', ' Finnish (fi)', ' French (fr)', ' German (de)', ' Greek (el)', ' Hebrew (he)', ' Hindi (hi)', ' Hungarian (hu)', ' Indonesian (id)', ' Italian (it)', ' Japanese (ja)', ' Korean (ko)', ' Norwegian (no)', ' Persian (fa)', ' Polish (pl)', ' Portuguese (pt)', ' Romanian (ro)', ' Russian (ru)', ' Spanish (es)', ' Swedish (sv)', ' Thai (th)', ' Turkish (tr)', ' Ukrainian (uk)', ' Vietnamese (vi)'])
    st.button('Continue', key='language_page_continue_button', on_click=to_language_analysis_page)

def by_language_root_concept_selection_page():
    st.header(f'Analyze by language: {state.language}')
    st.selectbox(label='Please select root concept:', key='root_concept_selection_box', options=root_synsets)
    st.button('Continue', key='language_analysis_page_continue_button', on_click=to_language_by_concept_page)

def by_language_sub_concept_selection_page():
    st.header(f'Analyze by language: {state.language}, root concept: {state.root_concept}')
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

    config = Config(width=400, height=400, directed=True, physics=True, hierarchical=True)

    state.concept = agraph(nodes=nodes, 
                        edges=edges, 
                        config=config)
    
    st.button('Continue', key='language_analysis_page_continue_button', on_click=to_language_by_concept_analysis_page)

def by_language_concept_analysis_page():
    st.header('Analysis')
    st.markdown(f'Analyzing: {state.concept} in captions in {state.language}')

    lang_code = state.language.split('(')[1].split(')')[0]
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
        image_file_paths = [f'/mnt/c/Users/uribe/PycharmProjects/datasets/crossmodal3600/images/{hex(iid)[2:].zfill(16)}.jpg' for iid in iids]
        images = []
        for file in image_file_paths:
            with open(file, "rb") as image:
                encoded = base64.b64encode(image.read()).decode()
                images.append(f"data:image/jpeg;base64,{encoded}")

        clicked = clickable_images(
            images,
            titles=[f"Image #{str(i)}" for i in range(len(image_file_paths))],
            div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
            img_style={"margin": "5px", "height": "200px"}
        )
        if clicked > -1:
            state.iid = iids[clicked]
            to_image_page()

def image_page():
    image_file_path = f'/mnt/c/Users/uribe/PycharmProjects/datasets/crossmodal3600/images/{hex(state.iid)[2:].zfill(16)}.jpg'
    st.image(image_file_path)

    st.subheader('Captions')
    lang_code = state.language.split('(')[1].split(')')[0]
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

print(f'cur_page: {state.cur_page}', flush=True)
if state.cur_page == 0:
    menu_page()
elif state.cur_page == 1:
    language_selection_page()
elif state.cur_page == 2:
    by_language_root_concept_selection_page()
elif state.cur_page == 3:
    by_language_sub_concept_selection_page()
elif state.cur_page == 4:
    by_language_concept_analysis_page()
elif state.cur_page == 5:
    image_page()
