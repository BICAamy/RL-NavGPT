import os
import json
import numpy as np

from action_plan_cache import attach_action_plans

def load_instr_datasets(anno_dir, dataset, splits):
    data = []
    for split in splits:
        filepath = os.path.join(anno_dir, f'{split}.json')
        with open(filepath) as f:
            new_data = json.load(f)

        data += new_data

    return data

def construct_instrs(anno_dir, dataset, splits, action_plan_cache=None):
    data = []
    if "instr" in splits[0]:
        data = load_instr_datasets(anno_dir, dataset, splits)
        if action_plan_cache:
            data = attach_action_plans(data, action_plan_cache)
        return data

    for i, item in enumerate(load_instr_datasets(anno_dir, dataset, splits)):
        # Split multiple instructions into separate entries 
        for j, instr in enumerate(item['instructions']):
            new_item = dict(item)
            new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
            new_item['instruction'] = instr
            del new_item['instructions']
            del new_item['instr_encodings']
            data.append(new_item)
    if action_plan_cache:
        data = attach_action_plans(data, action_plan_cache)
    return data
