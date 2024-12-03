import_datasets.py

# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import argparse
import os
import shutil
from os.path import exists

#from assistant_utils import process_assistant
from dialogflow_utils import process_dialogflow
#from mturk_utils import process_mturk

from nemo.utils import logging
from nemo.collections.nlp.data.data_utils import( #.datasets.datasets_utils import (
    DATABASE_EXISTS_TMP,
    MODE_EXISTS_TMP,
    create_dataset,
    get_dataset,
    if_exist,
    get_vocab,
)
#from nemo.collections.nlp.utils import get_vocab


def ids2text(ids, vocab):
    return ' '.join([vocab[int(id_)] for id_ in ids])


def process_atis(infold, outfold, modes=['train', 'test'], do_lower_case=False):
    """ MSFT's dataset, processed by Kaggle
    https://www.kaggle.com/siddhadev/atis-dataset-from-ms-cntk
    """
    vocab = get_vocab(f'{infold}/atis.dict.vocab.csv')

    if if_exist(outfold, [f'{mode}.tsv' for mode in modes]):
        logging.info(DATABASE_EXISTS_TMP.format('ATIS', outfold))
        return outfold
    logging.info(f'Processing ATIS dataset and storing at {outfold}.')

    os.makedirs(outfold, exist_ok=True)

    outfiles = {}
    for mode in modes:
        outfiles[mode] = open(os.path.join(outfold, mode + '.tsv'), 'w')
        outfiles[mode].write('sentence\tlabel\n')
        outfiles[mode + '_slots'] = open(f'{outfold}/{mode}_slots.tsv', 'w')

        queries = open(f'{infold}/atis.{mode}.query.csv', 'r').readlines()
        intents = open(f'{infold}/atis.{mode}.intent.csv', 'r').readlines()
        slots = open(f'{infold}/atis.{mode}.slots.csv', 'r').readlines()

        for i, query in enumerate(queries):
            sentence = ids2text(query.strip().split()[1:-1], vocab)
            if do_lower_case:
                sentence = sentence.lower()
            outfiles[mode].write(f'{sentence}\t{intents[i].strip()}\n')
            slot = ' '.join(slots[i].strip().split()[1:-1])
            outfiles[mode + '_slots'].write(slot + '\n')

    shutil.copyfile(f'{infold}/atis.dict.intent.csv', f'{outfold}/dict.intents.csv')
    shutil.copyfile(f'{infold}/atis.dict.slots.csv', f'{outfold}/dict.slots.csv')
    for mode in modes:
        outfiles[mode].close()


def process_snips(infold, outfold, do_lower_case, modes=['train', 'test'], dev_split=0.1):
    if not os.path.exists(infold):
        link = 'https://github.com/snipsco/spoken-language-understanding-research-datasets'
        raise ValueError(f'Data not found at {infold}. ' f'You may request to download the SNIPS dataset from {link}.')

    exist = True
    for dataset in ['light', 'speak', 'all']:
        if if_exist(f'{outfold}/{dataset}', [f'{mode}.tsv' for mode in modes]):
            logging.info(DATABASE_EXISTS_TMP.format('SNIPS-' + dataset, outfold))
        else:
            exist = False
    if exist:
        return outfold

    logging.info(f'Processing SNIPS dataset and storing at folders "speak", "light" and "all" under {outfold}.')
    logging.info(
        f'Processing and importing "smart-speaker-en-close-field" -> "speak" and "smart-speaker-en-close-field" -> "light".'
    )

    os.makedirs(outfold, exist_ok=True)

    speak_dir = 'smart-speaker-en-close-field'
    light_dir = 'smart-lights-en-close-field'

    light_files = [f'{infold}/{light_dir}/dataset.json']
    speak_files = [f'{infold}/{speak_dir}/training_dataset.json']
    speak_files.append(f'{infold}/{speak_dir}/test_dataset.json')

    light_train, light_dev, light_slots, light_intents = get_dataset(light_files, dev_split)
    speak_train, speak_dev, speak_slots, speak_intents = get_dataset(speak_files)

    create_dataset(light_train, light_dev, light_slots, light_intents, do_lower_case, f'{outfold}/light')
    create_dataset(speak_train, speak_dev, speak_slots, speak_intents, do_lower_case, f'{outfold}/speak')
    create_dataset(
        light_train + speak_train,
        light_dev + speak_dev,
        light_slots | speak_slots,
        light_intents | speak_intents,
        do_lower_case,
        f'{outfold}/all',
    )


def process_jarvis_datasets(
    infold, outfold, modes=['train', 'test', 'dev'], do_lower_case=False, ignore_prev_intent=False
):
    """ process and convert Jarvis datasets into NeMo's BIO format
    """
    dataset_name = "jarvis"
    if if_exist(outfold, ['dict.intents.csv', 'dict.slots.csv']):
        logging.info(DATABASE_EXISTS_TMP.format(dataset_name, outfold))
        return outfold

    logging.info(f'Processing {dataset_name} dataset and storing at {outfold}')

    os.makedirs(outfold, exist_ok=True)

    outfiles = {}
    intents_list = {}
    slots_list = {}
    slots_list_all = {}

    outfiles['dict_intents'] = open(f'{outfold}/dict.intents.csv', 'w')
    outfiles['dict_slots'] = open(f'{outfold}/dict.slots.csv', 'w')

    outfiles['dict_slots'].write('O\n')
    slots_list["O"] = 0
    slots_list_all["O"] = 0

    for mode in modes:
        if if_exist(outfold, [f'{mode}.tsv']):
            logging.info(MODE_EXISTS_TMP.format(mode, dataset_name, outfold, mode))
            continue

        if not if_exist(infold, [f'{mode}.tsv']):
            logging.info(f'{mode} mode of {dataset_name}' f' is skipped as it was not found.')
            continue

        outfiles[mode] = open(os.path.join(outfold, mode + '.tsv'), 'w')
        outfiles[mode].write('sentence\tlabel\n')
        outfiles[mode + '_slots'] = open(f'{outfold}/{mode}_slots.tsv', 'w')

        queries = open(f'{infold}/{mode}.tsv', 'r').readlines()

        for i, query in enumerate(queries):
            line_splits = query.strip().split("\t")
            if len(line_splits) == 3:
                intent_str, slot_tags_str, sentence = line_splits
            else:
                intent_str, sentence = line_splits
                slot_tags_str = ""

            if intent_str not in intents_list:
                intents_list[intent_str] = len(intents_list)
                outfiles['dict_intents'].write(f'{intent_str}\n')

            if ignore_prev_intent:
                start_token = 2
            else:
                start_token = 1

            if do_lower_case:
                sentence = sentence.lower()
            sentence_cld = " ".join(sentence.strip().split()[start_token:-1])
            outfiles[mode].write(f'{sentence_cld}\t' f'{str(intents_list[intent_str])}\n')

            slot_tags_list = []
            if slot_tags_str.strip():
                slot_tags = slot_tags_str.strip().split(",")
                for st in slot_tags:
                    if not st.strip():
                        continue
                    [start_i, end_i, slot_name] = st.strip().split(":")
                    slot_tags_list.append([int(start_i), int(end_i), slot_name])
                    if slot_name not in slots_list:
                        slots_list[slot_name] = len(slots_list)
                        slots_list_all[f'B-{slot_name}'] = len(slots_list_all)
                        slots_list_all[f'I-{slot_name}'] = len(slots_list_all)
                        outfiles['dict_slots'].write(f'B-{slot_name}\n')
                        outfiles['dict_slots'].write(f'I-{slot_name}\n')

            slot_tags_list.sort(key=lambda x: x[0])
            slots = []
            processed_index = 0
            for tag_start, tag_end, tag_str in slot_tags_list:
                if tag_start > processed_index:
                    words_list = sentence[processed_index:tag_start].strip().split()
                    slots.extend([str(slots_list_all['O'])] * len(words_list))
                words_list = sentence[tag_start:tag_end].strip().split()
                slots.append(str(slots_list_all[f'B-{tag_str}']))
                slots.extend([str(slots_list_all[f'I-{tag_str}'])] * (len(words_list) - 1))
                processed_index = tag_end

            if processed_index < len(sentence):
                words_list = sentence[processed_index:].strip().split()
                slots.extend([str(slots_list_all['O'])] * len(words_list))

            slots = slots[1:-1]
            slot = ' '.join(slots)
            outfiles[mode + '_slots'].write(slot + '\n')

        outfiles[mode + '_slots'].close()
        outfiles[mode].close()

    outfiles['dict_slots'].close()
    outfiles['dict_intents'].close()

    return outfold


if __name__ == "__main__":
    # Parse the command-line arguments.
    parser = argparse.ArgumentParser(description="Process and convert datasets into NeMo\'s format.")
    parser.add_argument(
        "--dataset_name",
        required=True,
        type=str,
        choices=['atis', 'snips', 'jarvis', 'dialogflow', 'mturk-processed', 'assistant'],
    )
    parser.add_argument(
        "--source_data_dir", required=True, type=str, help='path to the folder containing the dataset files'
    )
    parser.add_argument("--target_data_dir", required=True, type=str, help='path to save the processed dataset')
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument(
        "--ignore_prev_intent",
        action='store_true',
        help='ignores previous intent while importing datasets in jarvis\'s format',
    )
    parser.add_argument(
        "--use_full_dataset",
        action='store_true',
        help='using full 25K dataset versus smaller 10K version for assistant\'s dataset',
    )

    args = parser.parse_args()

    dataset_name = args.dataset_name
    source_dir = args.source_data_dir
    target_dir = args.target_data_dir

    if not exists(source_dir):
        raise FileNotFoundError(f"{source_dir} does not exist.")

    if dataset_name == 'atis':
        print('atis')
        #process_atis(infold=source_dir, outfold=target_dir, do_lower_case=args.do_lower_case)
    elif dataset_name == 'snips':
        process_snips(infold=source_dir, outfold=target_dir, do_lower_case=args.do_lower_case)
    elif dataset_name == 'jarvis':
        process_jarvis_datasets(
            infold=source_dir,
            outfold=target_dir,
            modes=["train", "test", "dev"],
            do_lower_case=args.do_lower_case,
            ignore_prev_intent=args.ignore_prev_intent,
        )
    elif dataset_name == 'dialogflow':
        process_dialogflow(infold=source_dir, outfold=target_dir)
    elif dataset_name == 'mturk-processed':
        print('murt process')
        #process_mturk(infold=source_dir, outfold=target_dir)
    elif dataset_name == 'assistant':
        print('assistant');
        #process_assistant(infold=source_dir, outfold=target_dir, use_full_dataset=args.use_full_dataset)
    else:
        raise ValueError(f'Dataset {dataset_name} is not supported.')


This script processes multiple datasets into NeMo's BIO format, which is widely used for tasks involving natural language understanding (NLU). Here’s a detailed breakdown of the code:

---

### **Imports**
- **`argparse`**: Parses command-line arguments.
- **`os`, `shutil`**: Manages file and directory operations.
- **`os.path.exists`**: Checks if a file or directory exists.

### **Utility Functions**
1. **`ids2text(ids, vocab)`**:
   - Converts a sequence of IDs into a string of tokens using a vocabulary (`vocab`).
   - Example: Transforms numerical IDs from datasets into human-readable sentences.

---

### **Dataset Processing Functions**

#### 1. **`process_atis()`**
   - Processes the **ATIS dataset**, a well-known dataset for intent classification and slot-filling.
   - Workflow:
     1. **Load Vocabulary**: Reads a vocabulary file.
     2. **Check if Dataset Exists**: If processed files already exist, skips processing.
     3. **Iterate over Modes**:
        - For each mode (`train`, `test`, etc.):
          - Read query, intent, and slot files.
          - Convert queries to text and write to `.tsv` files (e.g., `sentence\tintent_label`).
          - Write slot annotations to `_slots.tsv`.
     4. **Copy Supporting Files**: Copies intent and slot dictionaries to the target directory.

---

#### 2. **`process_snips()`**
   - Handles the **SNIPS dataset**, another dataset for intent and slot-filling tasks.
   - Features:
     - Splits the data into `train`, `test`, and optionally `dev`.
     - Processes multiple dataset versions (e.g., "light", "speak").
   - Workflow:
     1. Verifies dataset presence and skips processing if already done.
     2. Processes subsets like "light" and "speak".
     3. Combines datasets into an "all" version.

---

#### 3. **`process_jarvis_datasets()`**
   - Processes **Jarvis datasets** into NeMo's BIO format.
   - Features:
     - Handles multi-mode (`train`, `test`, `dev`) datasets.
     - Handles both intents and slot tags, including BIO format.
     - Supports case normalization and ignoring previous intents.
   - Workflow:
     1. Checks for processed files and skips if they exist.
     2. For each mode:
        - Reads queries, intents, and slots.
        - Maps intents to numerical IDs and slots to BIO-format tags.
        - Writes processed sentences and slot annotations.

---

### **Main Functionality**
- **Command-Line Arguments**:
  - **`--dataset_name`**: Dataset to process (e.g., `atis`, `snips`).
  - **`--source_data_dir`**: Directory of raw dataset files.
  - **`--target_data_dir`**: Directory to save processed files.
  - **`--do_lower_case`**: Optional flag to convert text to lowercase.
  - **`--ignore_prev_intent`**: For Jarvis datasets, skips intent history during processing.

- **Dataset Processing Dispatch**:
  - Based on the dataset name (`args.dataset_name`), the appropriate processing function is invoked.

---

### **Code Workflow**
1. Parse command-line arguments.
2. Verify the source directory exists.
3. Call the relevant dataset processing function:
   - **`process_atis`** for ATIS.
   - **`process_snips`** for SNIPS.
   - **`process_jarvis_datasets`** for Jarvis datasets.
   - **`process_dialogflow`** for Dialogflow datasets.
4. If an unsupported dataset is passed, raise an error.

---

### **Key Features**
1. **Extensibility**:
   - Supports additional datasets like Dialogflow and MTurk by defining their processing functions.
2. **Standardized Outputs**:
   - Creates consistent `.tsv` and dictionary files for easy use with downstream tasks.
3. **Custom Options**:
   - Options like `do_lower_case` and `ignore_prev_intent` allow for dataset-specific tweaks.

---

### **Example Usage**
```bash
python script.py --dataset_name atis \
                 --source_data_dir ./raw_data/atis \
                 --target_data_dir ./processed_data/atis \
                 --do_lower_case
```
This command processes the ATIS dataset from `./raw_data/atis`, converts all text to lowercase, and saves the output in `./processed_data/atis`.

---

### **Key Concepts**
- **BIO Format**: Common format for tagging tokens as **B**eginning, **I**nside, or **O**utside named entities.
- **Intent Classification**: Identifying the intent behind a user's input (e.g., "book flight").
- **Slot Filling**: Extracting entities from text (e.g., "Boston" as a `departure_city`).

This script is modular, extensible, and geared towards preparing NLU datasets for NeMo-based pipelines.

----
dialogflow_covertdata.py

# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import argparse
import os
import shutil
from os.path import exists

#from assistant_utils import process_assistant
from dialogflow_utils import process_dialogflow
#from mturk_utils import process_mturk

#from nemo import logging
#from nemo.collections.nlp.data.datasets.datasets_utils import (
#    DATABASE_EXISTS_TMP,
#    MODE_EXISTS_TMP,
#    create_dataset,
#    get_dataset,
#    if_exist,
#)
#from nemo.collections.nlp.utils import get_vocab


def ids2text(ids, vocab):
    return ' '.join([vocab[int(id_)] for id_ in ids])


def process_atis(infold, outfold, modes=['train', 'test'], do_lower_case=False):
    """ MSFT's dataset, processed by Kaggle
    https://www.kaggle.com/siddhadev/atis-dataset-from-ms-cntk
    """
    vocab = get_vocab(f'{infold}/atis.dict.vocab.csv')

    if if_exist(outfold, [f'{mode}.tsv' for mode in modes]):
        logging.info(DATABASE_EXISTS_TMP.format('ATIS', outfold))
        return outfold
    logging.info(f'Processing ATIS dataset and storing at {outfold}.')

    os.makedirs(outfold, exist_ok=True)

    outfiles = {}
    for mode in modes:
        outfiles[mode] = open(os.path.join(outfold, mode + '.tsv'), 'w')
        outfiles[mode].write('sentence\tlabel\n')
        outfiles[mode + '_slots'] = open(f'{outfold}/{mode}_slots.tsv', 'w')

        queries = open(f'{infold}/atis.{mode}.query.csv', 'r').readlines()
        intents = open(f'{infold}/atis.{mode}.intent.csv', 'r').readlines()
        slots = open(f'{infold}/atis.{mode}.slots.csv', 'r').readlines()

        for i, query in enumerate(queries):
            sentence = ids2text(query.strip().split()[1:-1], vocab)
            if do_lower_case:
                sentence = sentence.lower()
            outfiles[mode].write(f'{sentence}\t{intents[i].strip()}\n')
            slot = ' '.join(slots[i].strip().split()[1:-1])
            outfiles[mode + '_slots'].write(slot + '\n')

    shutil.copyfile(f'{infold}/atis.dict.intent.csv', f'{outfold}/dict.intents.csv')
    shutil.copyfile(f'{infold}/atis.dict.slots.csv', f'{outfold}/dict.slots.csv')
    for mode in modes:
        outfiles[mode].close()


def process_snips(infold, outfold, do_lower_case, modes=['train', 'test'], dev_split=0.1):
    if not os.path.exists(infold):
        link = 'https://github.com/snipsco/spoken-language-understanding-research-datasets'
        raise ValueError(f'Data not found at {infold}. ' f'You may request to download the SNIPS dataset from {link}.')

    exist = True
    for dataset in ['light', 'speak', 'all']:
        if if_exist(f'{outfold}/{dataset}', [f'{mode}.tsv' for mode in modes]):
            logging.info(DATABASE_EXISTS_TMP.format('SNIPS-' + dataset, outfold))
        else:
            exist = False
    if exist:
        return outfold

    logging.info(f'Processing SNIPS dataset and storing at folders "speak", "light" and "all" under {outfold}.')
    logging.info(
        f'Processing and importing "smart-speaker-en-close-field" -> "speak" and "smart-speaker-en-close-field" -> "light".'
    )

    os.makedirs(outfold, exist_ok=True)

    speak_dir = 'smart-speaker-en-close-field'
    light_dir = 'smart-lights-en-close-field'

    light_files = [f'{infold}/{light_dir}/dataset.json']
    speak_files = [f'{infold}/{speak_dir}/training_dataset.json']
    speak_files.append(f'{infold}/{speak_dir}/test_dataset.json')

    light_train, light_dev, light_slots, light_intents = get_dataset(light_files, dev_split)
    speak_train, speak_dev, speak_slots, speak_intents = get_dataset(speak_files)

    create_dataset(light_train, light_dev, light_slots, light_intents, do_lower_case, f'{outfold}/light')
    create_dataset(speak_train, speak_dev, speak_slots, speak_intents, do_lower_case, f'{outfold}/speak')
    create_dataset(
        light_train + speak_train,
        light_dev + speak_dev,
        light_slots | speak_slots,
        light_intents | speak_intents,
        do_lower_case,
        f'{outfold}/all',
    )


def process_jarvis_datasets(
    infold, outfold, modes=['train', 'test', 'dev'], do_lower_case=False, ignore_prev_intent=False
):
    """ process and convert Jarvis datasets into NeMo's BIO format
    """
    dataset_name = "jarvis"
    if if_exist(outfold, ['dict.intents.csv', 'dict.slots.csv']):
        logging.info(DATABASE_EXISTS_TMP.format(dataset_name, outfold))
        return outfold

    logging.info(f'Processing {dataset_name} dataset and storing at {outfold}')

    os.makedirs(outfold, exist_ok=True)

    outfiles = {}
    intents_list = {}
    slots_list = {}
    slots_list_all = {}

    outfiles['dict_intents'] = open(f'{outfold}/dict.intents.csv', 'w')
    outfiles['dict_slots'] = open(f'{outfold}/dict.slots.csv', 'w')

    outfiles['dict_slots'].write('O\n')
    slots_list["O"] = 0
    slots_list_all["O"] = 0

    for mode in modes:
        if if_exist(outfold, [f'{mode}.tsv']):
            logging.info(MODE_EXISTS_TMP.format(mode, dataset_name, outfold, mode))
            continue

        if not if_exist(infold, [f'{mode}.tsv']):
            logging.info(f'{mode} mode of {dataset_name}' f' is skipped as it was not found.')
            continue

        outfiles[mode] = open(os.path.join(outfold, mode + '.tsv'), 'w')
        outfiles[mode].write('sentence\tlabel\n')
        outfiles[mode + '_slots'] = open(f'{outfold}/{mode}_slots.tsv', 'w')

        queries = open(f'{infold}/{mode}.tsv', 'r').readlines()

        for i, query in enumerate(queries):
            line_splits = query.strip().split("\t")
            if len(line_splits) == 3:
                intent_str, slot_tags_str, sentence = line_splits
            else:
                intent_str, sentence = line_splits
                slot_tags_str = ""

            if intent_str not in intents_list:
                intents_list[intent_str] = len(intents_list)
                outfiles['dict_intents'].write(f'{intent_str}\n')

            if ignore_prev_intent:
                start_token = 2
            else:
                start_token = 1

            if do_lower_case:
                sentence = sentence.lower()
            sentence_cld = " ".join(sentence.strip().split()[start_token:-1])
            outfiles[mode].write(f'{sentence_cld}\t' f'{str(intents_list[intent_str])}\n')

            slot_tags_list = []
            if slot_tags_str.strip():
                slot_tags = slot_tags_str.strip().split(",")
                for st in slot_tags:
                    if not st.strip():
                        continue
                    [start_i, end_i, slot_name] = st.strip().split(":")
                    slot_tags_list.append([int(start_i), int(end_i), slot_name])
                    if slot_name not in slots_list:
                        slots_list[slot_name] = len(slots_list)
                        slots_list_all[f'B-{slot_name}'] = len(slots_list_all)
                        slots_list_all[f'I-{slot_name}'] = len(slots_list_all)
                        outfiles['dict_slots'].write(f'B-{slot_name}\n')
                        outfiles['dict_slots'].write(f'I-{slot_name}\n')

            slot_tags_list.sort(key=lambda x: x[0])
            slots = []
            processed_index = 0
            for tag_start, tag_end, tag_str in slot_tags_list:
                if tag_start > processed_index:
                    words_list = sentence[processed_index:tag_start].strip().split()
                    slots.extend([str(slots_list_all['O'])] * len(words_list))
                words_list = sentence[tag_start:tag_end].strip().split()
                slots.append(str(slots_list_all[f'B-{tag_str}']))
                slots.extend([str(slots_list_all[f'I-{tag_str}'])] * (len(words_list) - 1))
                processed_index = tag_end

            if processed_index < len(sentence):
                words_list = sentence[processed_index:].strip().split()
                slots.extend([str(slots_list_all['O'])] * len(words_list))

            slots = slots[1:-1]
            slot = ' '.join(slots)
            outfiles[mode + '_slots'].write(slot + '\n')

        outfiles[mode + '_slots'].close()
        outfiles[mode].close()

    outfiles['dict_slots'].close()
    outfiles['dict_intents'].close()

    return outfold


if __name__ == "__main__":
    # Parse the command-line arguments.
    parser = argparse.ArgumentParser(description="Process and convert datasets into NeMo\'s format.")
    parser.add_argument(
        "--dataset_name",
        required=True,
        type=str,
        choices=['atis', 'snips', 'jarvis', 'dialogflow', 'mturk-processed', 'assistant'],
    )
    parser.add_argument(
        "--source_data_dir", required=True, type=str, help='path to the folder containing the dataset files'
    )
    parser.add_argument("--target_data_dir", required=True, type=str, help='path to save the processed dataset')
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument(
        "--ignore_prev_intent",
        action='store_true',
        help='ignores previous intent while importing datasets in jarvis\'s format',
    )
    parser.add_argument(
        "--use_full_dataset",
        action='store_true',
        help='using full 25K dataset versus smaller 10K version for assistant\'s dataset',
    )

    args = parser.parse_args()

    dataset_name = args.dataset_name
    source_dir = args.source_data_dir
    target_dir = args.target_data_dir

    if not exists(source_dir):
        raise FileNotFoundError(f"{source_dir} does not exist.")

    if dataset_name == 'atis':
        process_atis(infold=source_dir, outfold=target_dir, do_lower_case=args.do_lower_case)
    elif dataset_name == 'snips':
        process_snips(infold=source_dir, outfold=target_dir, do_lower_case=args.do_lower_case)
    elif dataset_name == 'jarvis':
        process_jarvis_datasets(
            infold=source_dir,
            outfold=target_dir,
            modes=["train", "test", "dev"],
            do_lower_case=args.do_lower_case,
            ignore_prev_intent=args.ignore_prev_intent,
        )
    elif dataset_name == 'dialogflow':
        process_dialogflow(infold=source_dir, outfold=target_dir)
    #elif dataset_name == 'mturk-processed':
    #    process_mturk(infold=source_dir, outfold=target_dir)
    #elif dataset_name == 'assistant':
     #   process_assistant(infold=source_dir, outfold=target_dir, use_full_dataset=args.use_full_dataset)
    else:
        raise ValueError(f'Dataset {dataset_name} is not supported.')


Here's a detailed breakdown of the code provided:

---

### **Imports and Setup**
The script imports several libraries and utility functions to handle file I/O, logging, argument parsing, and dataset-specific processing:

```python
import argparse
import os
import shutil
from os.path import exists
```
- **`argparse`**: Handles command-line arguments.
- **`os` & `shutil`**: Handle file operations (creating directories, copying files, etc.).
- **`exists`**: Checks if a file or directory exists.

#### NeMo and Utility Imports:
```python
from nemo.utils import logging
from nemo.collections.nlp.data.data_utils import(
    DATABASE_EXISTS_TMP,
    MODE_EXISTS_TMP,
    create_dataset,
    get_dataset,
    if_exist,
    get_vocab,
)
```
- **NeMo**: A toolkit for building AI models, including dataset utilities like:
  - `get_vocab()`: Loads vocabulary from a file.
  - `if_exist()`: Checks if specific files exist in a directory.
  - `create_dataset()` & `get_dataset()`: Helpers for creating and splitting datasets.
- **Logging**: Used for structured logging throughout the script.

---

### **Utility Function: `ids2text`**
```python
def ids2text(ids, vocab):
    return ' '.join([vocab[int(id_)] for id_ in ids])
```
- Converts a list of numeric IDs into human-readable text using a vocabulary mapping.
- **Input**:
  - `ids`: List of IDs.
  - `vocab`: Mapping of ID to token.
- **Output**: A single string of joined tokens.

---

### **Dataset Processing Functions**

#### **1. `process_atis`**
```python
def process_atis(infold, outfold, modes=['train', 'test'], do_lower_case=False):
```
Processes the **ATIS dataset** from CSV files. Key steps:
1. Load vocabulary from `atis.dict.vocab.csv`.
2. Check if the processed dataset already exists. If yes, log and return.
3. Create output directory if it doesn’t exist.
4. Process files for specified modes (`train` and `test`):
   - Convert queries (IDs) into text using `ids2text`.
   - Write processed queries and their labels to `.tsv` files.
   - Save slot information separately.
5. Copy the intent and slot mappings from input to output.

---

#### **2. `process_snips`**
```python
def process_snips(infold, outfold, do_lower_case, modes=['train', 'test'], dev_split=0.1):
```
Processes the **SNIPS dataset**, available in JSON format. Key steps:
1. Validate the existence of input files and throw an error if missing.
2. Check if processed files already exist for various configurations (`light`, `speak`, `all`).
3. Process datasets for different sub-domains:
   - **Light and Speak**:
     - Load JSON files for training/testing.
     - Split training into train/dev sets based on `dev_split`.
   - **Combined Dataset**:
     - Merge datasets from both sub-domains into a single dataset.
4. Call `create_dataset()` for each processed set to generate `.tsv` files.

---

#### **3. `process_jarvis_datasets`**
```python
def process_jarvis_datasets(infold, outfold, modes=['train', 'test', 'dev'], do_lower_case=False, ignore_prev_intent=False):
```
Processes datasets in the **Jarvis format** (NeMo's BIO format). Key steps:
1. Check if intent and slot mappings (`dict.intents.csv`, `dict.slots.csv`) exist.
2. Create mappings for intents and slots dynamically while processing queries.
3. Process each mode (`train`, `test`, `dev`):
   - Parse queries into sentences, intents, and slots.
   - Map intents and slots to their IDs and write to `.tsv` files.
4. Handle slot tagging using BIO format for named entities.

---

### **Main Script Logic**
```python
if __name__ == "__main__":
```
The script is executed as a standalone program:
1. Parse command-line arguments to specify:
   - Dataset type (`--dataset_name`).
   - Input/output directories.
   - Flags for case normalization, intent handling, etc.
2. Validate the existence of the input directory.
3. Call the appropriate dataset processing function based on `--dataset_name`.
   - If the dataset isn't supported, raise an error.

---

### **Command-Line Arguments**
```python
parser = argparse.ArgumentParser(description="Process and convert datasets into NeMo\'s format.")
parser.add_argument("--dataset_name", required=True, type=str, choices=[...])
parser.add_argument("--source_data_dir", required=True, type=str)
parser.add_argument("--target_data_dir", required=True, type=str)
parser.add_argument("--do_lower_case", action='store_true')
parser.add_argument("--ignore_prev_intent", action='store_true')
parser.add_argument("--use_full_dataset", action='store_true')
```
- Defines the expected command-line arguments:
  - `--dataset_name`: Type of dataset to process.
  - `--source_data_dir`: Input directory.
  - `--target_data_dir`: Output directory.
  - Flags for additional processing options.

---

### **Conclusion**
This script provides a modular framework for processing datasets into NeMo's format. It supports:
- **ATIS**, **SNIPS**, **Jarvis**, and other datasets.
- Dynamic intent/slot mapping.
- Conversion to `.tsv` format for training/testing.
- Options for lowercasing and handling special cases.
Sure! Let me expand on each part of the code to provide a more comprehensive understanding of its components and functionality. This way, you'll have a clearer picture of how the script works and what each section does.

---

### **Purpose of the Script**
The script is designed to preprocess various natural language understanding (NLU) datasets into a specific format compatible with **NeMo**, NVIDIA's AI development toolkit. It handles datasets like ATIS, SNIPS, Jarvis, and others, converting them into a consistent format for tasks such as intent classification and slot-filling.

---

### **1. Imports and Their Roles**

#### **General Libraries**
```python
import argparse
import os
import shutil
from os.path import exists
```
- **`argparse`**: Handles command-line arguments to make the script flexible and reusable.
- **`os` & `shutil`**: Provides tools to interact with the file system:
  - Create directories (`os.makedirs()`).
  - Check for file existence (`os.path.exists()`).
  - Copy files (`shutil.copyfile()`).

#### **Domain-Specific Imports**
```python
from nemo.utils import logging
from nemo.collections.nlp.data.data_utils import(
    DATABASE_EXISTS_TMP,
    MODE_EXISTS_TMP,
    create_dataset,
    get_dataset,
    if_exist,
    get_vocab,
)
```
- **`logging`**: Provides structured logging messages for debugging and progress tracking.
- **NeMo utilities**:
  - **`DATABASE_EXISTS_TMP` and `MODE_EXISTS_TMP`**: Template strings for log messages indicating dataset existence.
  - **`create_dataset()`**: Utility to split, preprocess, and save datasets.
  - **`get_dataset()`**: Reads and parses datasets, splitting them into train/dev/test subsets.
  - **`if_exist()`**: Checks for the existence of required output files.
  - **`get_vocab()`**: Loads a vocabulary file for token-ID mapping.

---

### **2. Utility Functions**

#### **`ids2text`**
```python
def ids2text(ids, vocab):
    return ' '.join([vocab[int(id_)] for id_ in ids])
```
- **Purpose**: Converts a sequence of numeric token IDs into a human-readable sentence using the vocabulary file.
- **How It Works**:
  1. Iterate over the list of token IDs (`ids`).
  2. Use the vocabulary (`vocab`) to map each ID to its corresponding word.
  3. Join the words with spaces to form a sentence.

- **Example**:
  - Input: `ids = [1, 4, 7]`, `vocab = {1: 'book', 4: 'a', 7: 'flight'}`
  - Output: `"book a flight"`

---

### **3. Dataset-Specific Processing Functions**

#### **a) `process_atis`**
Processes the **Airline Travel Information System (ATIS)** dataset. This dataset includes:
- Queries in the form of token IDs.
- Intent labels (e.g., "BookFlight").
- Slot tags for token-level annotations (e.g., "B-Destination").

```python
def process_atis(infold, outfold, modes=['train', 'test'], do_lower_case=False):
```
- **Steps**:
  1. **Load Vocabulary**:
     ```python
     vocab = get_vocab(f'{infold}/atis.dict.vocab.csv')
     ```
     Load a CSV file mapping token IDs to words.
  
  2. **Check Output Existence**:
     ```python
     if if_exist(outfold, [f'{mode}.tsv' for mode in modes]):
         logging.info(DATABASE_EXISTS_TMP.format('ATIS', outfold))
         return outfold
     ```
     Skip processing if the required `.tsv` files already exist in the output folder.

  3. **Process Queries**:
     For each mode (`train`, `test`):
     - Convert token IDs into sentences.
     - Match sentences with their intent labels and slot tags.
     - Save this data in `.tsv` format.

  4. **Copy Metadata**:
     ```python
     shutil.copyfile(f'{infold}/atis.dict.intent.csv', f'{outfold}/dict.intents.csv')
     ```
     Copy intent and slot mappings to the output folder.

- **Output Files**:
  - `train.tsv` and `test.tsv`: Processed queries with labels.
  - `train_slots.tsv` and `test_slots.tsv`: Slot tags for queries.

---

#### **b) `process_snips`**
Handles the **SNIPS dataset**, which is structured differently (JSON files).

```python
def process_snips(infold, outfold, do_lower_case, modes=['train', 'test'], dev_split=0.1):
```
- **Purpose**: Processes SNIPS datasets from JSON format into `.tsv` files.

- **Steps**:
  1. **Validation**:
     ```python
     if not os.path.exists(infold):
         raise ValueError(f'Data not found at {infold}.')
     ```
     Ensures the input folder exists.

  2. **Check Existing Outputs**:
     ```python
     exist = True
     for dataset in ['light', 'speak', 'all']:
         if if_exist(f'{outfold}/{dataset}', [f'{mode}.tsv' for mode in modes]):
             ...
     ```
     If outputs exist for sub-domains (`light`, `speak`, and combined), skip processing.

  3. **Processing Sub-Domains**:
     - Use `get_dataset()` to parse the JSON files for training/testing datasets.
     - Split training data into train/dev subsets using `dev_split`.

  4. **Generate `.tsv` Files**:
     Call `create_dataset()` for each processed dataset:
     - **Light**: Processes data from a specific sub-domain.
     - **Speak**: Processes data from another sub-domain.
     - **All**: Combines the above datasets.

---

#### **c) `process_jarvis_datasets`**
Converts datasets into the **BIO format**, used for slot tagging tasks.

```python
def process_jarvis_datasets(infold, outfold, modes=['train', 'test', 'dev'], do_lower_case=False, ignore_prev_intent=False):
```
- **Purpose**: Processes and standardizes datasets into the format required by NVIDIA's Jarvis model.

- **Key Features**:
  - Supports BIO tagging for slot annotations:
    - `B-<slot>`: Beginning of a slot.
    - `I-<slot>`: Inside a slot.
    - `O`: Outside any slot.

- **Steps**:
  1. **Create Slot and Intent Mappings**:
     ```python
     intents_list = {}
     slots_list = {}
     slots_list_all = {'O': 0}
     ```
     Build mappings for intents and slots dynamically while processing queries.

  2. **Process Sentences and Tags**:
     ```python
     if len(line_splits) == 3:
         intent_str, slot_tags_str, sentence = line_splits
     ```
     Parse each query line into its components (intent, slots, sentence).

  3. **Handle BIO Tagging**:
     ```python
     slots.extend([str(slots_list_all['O'])] * len(words_list))
     ```
     Add BIO tags for each token in the sentence.

  4. **Save Outputs**:
     Write processed data to `.tsv` files for train, test, and dev subsets.

---

### **4. Main Script Logic**
The script uses `argparse` to accept command-line arguments for specifying dataset type, input/output directories, and processing options.

```python
if __name__ == "__main__":
```
- Parses arguments and validates inputs:
  ```python
  parser.add_argument("--dataset_name", required=True, type=str, ...)
  if not exists(source_dir):
      raise FileNotFoundError(f"{source_dir} does not exist.")
  ```

- Calls the appropriate dataset-specific function based on the `dataset_name` argument:
  ```python
  if dataset_name == 'atis':
      process_atis(infold=source_dir, outfold=target_dir, do_lower_case=args.do_lower_case)
  ```

---

### **How to Run the Script**
- Command-line Example:
  ```bash
  python preprocess_datasets.py \
      --dataset_name atis \
      --source_data_dir ./raw_data/atis \
      --target_data_dir ./processed_data/atis \
      --do_lower_case
  ```
  This processes the ATIS dataset with lowercased text.

---

### **Key Takeaways**
- **Modularity**: Each dataset type has a dedicated processing function.
- **Flexibility**: Command-line arguments make it reusable across datasets.
- **Standardization**: Converts diverse datasets into NeMo-compatible `.tsv` files.
- **Scalability**: Supports multiple datasets with minimal changes.
  ----------------------

  intent_slot_classification_config.yaml

  
