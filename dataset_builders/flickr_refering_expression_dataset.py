import os
import time
from collections import defaultdict
from xml.dom import minidom

class FlickrReferingExpressionDatasetBuilder:
    def __init__(self, root):
        self.root = root
        self.image_dir = os.path.join(root, 'images')
        annotations_dir = os.path.join(root, 'annotations')
        self.bbox_dir = os.path.join(annotations_dir, 'Annotations')
        self.ref_exp_dir = os.path.join(annotations_dir, 'Sentences')
        captions_dir = os.path.join(root, 'tokens')
        self.captions_file = os.path.join(captions_dir, 'results_20130124.token')

        self.coord_strs = ['xmin', 'ymin', 'xmax', 'ymax']
        self.coord_str_to_ind = {self.coord_strs[x]: x for x in range(len(self.coord_strs))}

    def get_image_id_to_captions(self):
        print('Collecting image id to captions...')
        image_id_to_captions = defaultdict(list)
        with open(self.captions_file, encoding='utf-8') as fp:
            for line in fp:
                split_line = line.strip().split('g#')
                img_file_name = split_line[0] + 'g'
                image_id = int(img_file_name.split('.')[0])
                caption_info = split_line[1].split('\t')
                caption = caption_info[1]  # The first token is caption number
                caption_ind = int(caption_info[0])
                
                image_id_to_captions[image_id].append({'caption': caption, 'caption_id': caption_ind})

        return image_id_to_captions

    def get_image_id_to_ref_exps(self):
        print('Collecting image id to chain...')
        count = 0
        t = time.time()
        image_id_to_chains = {}
        bbox_files = os.listdir(self.bbox_dir)
        for image_bbox_file in bbox_files:
            if count % 1000 == 0:
                print('\tStarting ' + str(count) + ' out of ' + str(len(bbox_files)) + ', time from prev ' + str(time.time()-t), flush=True)
                t = time.time()
            count += 1
            image_id = int(image_bbox_file.split('.')[0])
            if image_id not in image_id_to_chains:
                image_id_to_chains[image_id] = {}
            xml_filepath = os.path.join(self.bbox_dir, image_bbox_file)
            xml_doc = minidom.parse(xml_filepath)
            for child_node in xml_doc.childNodes[0].childNodes:
                # The bounding boxes are located inside a node named "object"
                if child_node.nodeName == u'object':
                    # Go over all of the children of this node: if we find bndbox, this object is a bounding box
                    box_chain = None
                    for inner_child_node in child_node.childNodes:
                        if inner_child_node.nodeName == u'name':
                            box_chain = int(inner_child_node.childNodes[0].data)
                        if inner_child_node.nodeName == u'bndbox':
                            bounding_box = [None, None, None, None]
                            for val_node in inner_child_node.childNodes:
                                node_name = val_node.nodeName
                                if node_name in self.coord_strs:
                                    coord_ind = self.coord_str_to_ind[node_name]
                                    bounding_box[coord_ind] = int(val_node.childNodes[0].data)

                            # Check that all coordinates were found
                            none_inds = [x for x in range(len(bounding_box)) if x is None]
                            assert len(none_inds) == 0
                            assert box_chain is not None

                            # Document chain
                            if box_chain not in image_id_to_chains[image_id]:
                                image_id_to_chains[image_id][box_chain] = []
                            image_id_to_chains[image_id][box_chain].append(bounding_box)

        print('Collecting image id to ref exp...')
        count = 0
        t = time.time()
        image_id_to_ref_exp = defaultdict(list)
        ref_exp_files = os.listdir(self.ref_exp_dir)
        for image_ref_exp_file in ref_exp_files:
            if count % 1000 == 0:
                print('\tStarting ' + str(count) + ' out of ' + str(len(ref_exp_files)) + ', time from prev ' + str(time.time()-t), flush=True)
                t = time.time()
            count += 1
            image_id = int(image_ref_exp_file.split('.')[0])
            ref_exps = []
            with open(os.path.join(self.ref_exp_dir, image_ref_exp_file)) as fp:
                for line in fp:
                    if len(line.strip()) == 0:
                        continue
                    ref_exps.append([])
                    line_parts = line.split('[/EN#')
                    for part in line_parts[1:]:
                        ref_exp = part.split(']')[0]
                        ref_exp_parts = ref_exp.split()
                        text = ' '.join(ref_exp_parts[1:])
                        chain_id_and_types = ref_exp_parts[0].split('/')
                        chain_id = chain_id_and_types[0]
                        types = chain_id_and_types[1:]
                        ref_exps[-1].append({'text': text, 'chain_id': chain_id, 'types': types})
            image_id_to_ref_exp[image_id].append(ref_exps)

        return image_id_to_chains, image_id_to_ref_exp
    
    def build_dataset(self):
        image_id_to_captions = self.get_image_id_to_captions()
        image_id_to_chains, image_id_to_ref_exps = self.get_image_id_to_ref_exps()
        return {'captions': image_id_to_captions, 'chains': image_id_to_chains, 'ref_exps': image_id_to_ref_exps}
