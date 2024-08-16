# Cross-Lingual and Cross-Cultural Variation in Image Descriptions

## Introduction

Do speakers of different languages talk differently about images? We analyze the CrossModal3600 dataset and develop a method to accurately identify entities mentioned in captions and present in the images.
We publish our object mentions dataset for future use, as well as our code for entity identification.

## Object mentions dataset

Our dataset files are all under the datasets folder. Each file corresponds to a single language. The language codes are:

Arabic (ar), Chinese-Simplified (zh), Croatian (hr), Czech (cs), Danish (da), Dutch (nl), English (en), Filipino (fil), Finnish (fi), French (fr), German (de), Greek (el), Hebrew (he), Hindi (hi), Hungarian (hu), Indonesian (id), Italian (it), Japanese (ja), Korean (ko), Norwegian (no), Persian (fa), Polish (pl), Portuguese (pt), Romanian (ro), Russian (ru), Spanish (es), Swedish (sv), Thai (th), Turkish (tr), Ukrainian (uk), Vietnamese (vi)

We advise against using the five additional low-resource languages due to the poor quality of their translations to English, which is a crucial step in our pipeline:

Bengali (bn), Maori (mi), Cusco Quechua (quz), Swahili (sw), Telugu (te)

Each language's dataset is a json file containing a list of samples, where each sample has the following fields:

- image_id: the image id. In CrossModal3600 each image's file name is a string representing a hexadecimal number, and we use the decimal version of this number as the image id. To get the file name from the image id use ```hex(image_id)[2:].zfill(16) + '.jpg'```.
- caption: the translated caption, or the original caption in case the language of this dataset is English.
- orig: only in non-English datasets, the caption in the original language.
- source: the name of the dataset.
- synsets: the identified list of synsets. Each element in this list is a tuple of representing a phrase identified as a synset, with the following fields: ```(<phrase starting token index>, <phrase ending token index>, <phrase string>, <synset>, <depth from synset>)```. Some phrases are identified as synsets which are descendents of one of our predefined synsets, in this case the ```depth from synset``` is the vertical distance between the identified synset and the sysnet mentioned in the tuple.

### Synset filtering (available only for CrossModal3600)

We manually annotated the root synsets instantiated in each image in the CrossModal3600 dataset (this annotation is available under data/xm3600_annotation.csv). For a more accurate corpus, we recommend to filter out synsets for which the corresponding root synset is not instantiated in the image. This can be done using our ```verify_synset_in_image``` method:

```
from utils import get_image_id_to_root_synsets, verify_synset_in_image
from datasets import get_processed_dataset

iid2root_synset = get_image_id_to_root_synsets() # Loads the annotation data from data/xm3600_annotation.csv
data = get_processed_dataset('xm3600_he') # Loads the CrossModal3600 Hebrew dataset from the datasets/xm3600_he.json file
filtered_data = []
for sample in data:
    sample['synsets'] = [synset_tuple for synset_tuple in sample['synsets'] if verify_synset_in_image(synset_tuple[3], sample['image_id'], iid2root_synset)]
    filtered_data.append(sample)
```

## Using our code

### Installation
Python version: 3.9.2
```
pip install -r requirements.txt
```

### Download translated datasets
Go to [this link](https://drive.google.com/drive/folders/1JtpCaGhFh30rpX8pfvFti0dlzChWceVJ?usp=sharing) and download the xm3600 folder. Place the folder under the project root. In case you want to use STAIR-captions, download both the original and the translated json files, and update their path in the config.py file.

### Create the object mentions dataset
Run:
```
python src/process_dataset.py <dataset name>
```
Currently available datasets are CrossModal3600 datasets (xm3600 + language code, e.g. ```xm3600_he```), COCO, STAIR-captions.
To add a new dataset to be processed, add a relevant if clause in the get_orig_dataset method in the get_dataset.py file.
