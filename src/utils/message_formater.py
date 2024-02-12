#!/usr/bin/env python
# encoding: utf-8
"""
format input messages, global_arguments, tools, tool_choice to a input string
"""
from typing import List, Dict, Optional, Union
import json
from io import StringIO
from ruamel import yaml
from ruamel.yaml.comments import CommentedMap as OrderedDict
import math
import numpy as np



folded = yaml.scalarstring.FoldedScalarString
literal = yaml.scalarstring.LiteralScalarString
double_quoted = yaml.scalarstring.DoubleQuotedScalarString
commented_map = yaml.comments.CommentedSeq
threshold = 1000


class EmptyValueException(Exception):
    pass


def custom_yaml_dump(item):
    #if item is None:
    #    raise EmptyValueException
    #elif isinstance(item, dict):
    if isinstance(item, dict):
        data = OrderedDict()
        for key in item.keys():
            #if item[key] is None:
            #    raise EmptyValueException
            if key == 'object':
                data[key] = item[key]
            elif key == 'name':
                data[key] = double_quoted(item[key])
            elif key == 'arguments':
                data[key] = custom_yaml_dump(item[key])
            elif key == 'plan' and item[key] == '':
                data[key] = []
            elif (key == 'code' or key == 'content' or key == 'chat' or key == 'command') and isinstance(item[key], str):
                data[key] = without_quoted_dump(item[key])
            elif isinstance(item[key], list):
                data[key] = list_dump(item[key])
            else:
                data[key] = custom_yaml_dump(item[key])
        return data
    elif isinstance(item, str):
        length = len(item)
        if length > threshold:
            return literal(item)
        elif length > 100:
            prob = math.atan(length / threshold) / (math.pi / 2)
            if np.random.rand() < prob:
                return literal(item)
            else:
                return double_quoted(item)
        else:
            return double_quoted(item)
    elif isinstance(item, list):
        item = [custom_yaml_dump(x) for x in item]
        return item
    elif isinstance(item, (int,float)):
        return item
    elif item is None:
        return item
    else:
        raise NotImplementedError(f"type {type(item)} is not supported.")


def list_dump(item):
    for i in range(len(item)):
        if isinstance(item[i],str):
            item[i] = double_quoted(item[i])
        elif isinstance(item[i], int):
            item[i] = item[i]
        else:
            item[i] = custom_yaml_dump(item[i])
    try:
        if len(item) > 0 and ((isinstance(item[0], str) and sum(len(str(i)) for i in item) < 100) or isinstance(item[0], int) or isinstance(item[0], float)):
            item = commented_map(item, lcs=True)
            item.fa.set_flow_style()
    except:
        pass
    return item


def without_quoted_dump(item):
    return literal(item)


def yaml_load(string):
    """load yaml"""
    f = StringIO(string)
    yaml_obj = yaml.YAML()
    yaml_obj.width = 1000000
    data = yaml_obj.load(f)
    return data


def yaml_load_all(string):
    """load yaml"""
    f = StringIO(string)
    yaml_obj = yaml.YAML()
    yaml_obj.width = 1000000
    data = yaml_obj.load_all(f)
    return data


def yaml_dump(item):
    f = StringIO()
    try:
        item = custom_yaml_dump(item)
        yaml_obj = yaml.YAML()
        yaml_obj.width = 1000000
        yaml_obj.dump(item, f)
    except EmptyValueException:
        return None
    f.seek(0)
    string = f.read()
    return string


def yaml_origin_dump(item):
    f = StringIO()
    try:
        yaml_obj = yaml.YAML()
        yaml_obj.width = 1000000
        yaml_obj.dump(item, f)
    except EmptyValueException:
        return None
    f.seek(0)
    string = f.read()
    return string


def yaml_dump_all(item):
    f = StringIO()
    try:
        item = custom_yaml_dump(item)
        yaml_obj = yaml.YAML()
        yaml_obj.width = 1000000
        yaml_obj.dump_all(item, f)
    except EmptyValueException:
        return None
    f.seek(0)
    string = f.read()
    return string

def yaml_origin_dump_all(item):
    f = StringIO()
    try:
        yaml_obj = yaml.YAML()
        yaml_obj.width = 1000000
        yaml_obj.dump_all(item, f)
    except EmptyValueException:
        return None
    f.seek(0)
    string = f.read()
    return string


def custom_json_dump(item):
    if item is None:
        raise EmptyValueException
    elif isinstance(item, dict):
        data = OrderedDict()
        for key in item.keys():
            if item[key] is None:
                raise EmptyValueException
            if key == 'global_arguments' or key == 'tool_choice':
                data[key] = custom_json_dump(item[key])
            else:
                data[key] = item[key]
        return data
    elif isinstance(item, str) and '\n' in item:
        return literal(item)
    else:
        return item


def json_dump(item):
    try:
        item = custom_json_dump(item)
        string = json.dumps(item, ensure_ascii=False)
    except EmptyValueException:
        return None
    return string


def message_format(msg, config):
    """https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L343"""
    if msg["role"].lower().startswith("user"):
        string = config['user_template'].format(content=msg['content'])
    elif msg["role"].lower().startswith("system"):
        string = config['user_template'].format(content=msg['content'])
    elif msg["role"].lower().startswith("assistant") or msg["role"].lower().startswith("agent") :
        if "content" in msg:
            string = config['assistant_template'].format(content=msg['content'])
        else:
            string = ""
        if "object_calls" in msg:
            if config['output_dump_method'] == 'yaml':
                string += "\n" + yaml_dump_all(msg["object_calls"]).strip()
            elif config['output_dump_method'] == 'yaml_origin':
                string += "\n" + yaml_origin_dump_all(msg["object_calls"]).strip()
            elif config['output_dump_method'] == 'json':
                string += "\n" + json.dumps(msg["object_calls"], ensure_ascii=False).strip()
            else:
                raise NotImplementedError
    elif msg["role"].lower().startswith("tool"):
        string = config['tool_response_template'].format(content=msg['content'])
    else:
        print(msg["role"])
        #raise NotImplementedError
        return None
    return string.strip()


def merge_messages(messages):
    """convert system message to user message, merge continous user message into one user message"""
    new_messages = []
    pre_role = ""
    for msg in messages:
        # system message should be merged with user message
        # reference: https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L324
        if msg['role'] == 'system':
            role = 'user'
            content = msg["content"]
        else:
            role = msg['role']
            content = msg['content']

        if role == pre_role:
            new_messages[-1]["content"] += "\n" + content
        else:
            new_messages.append({'role': role, 'content': content})
        pre_role = role
    # logger.info(f"merge {len(messages)} {len(new_messages)} messages")
    return new_messages


def find_system_msg(messages):
    """find system message position"""
    idx = -1
    for i, msg in enumerate(messages):
        if msg["role"] == "system":
            idx = i
    return idx


def my_dump(item, dump_method):
    """my dump method, yaml or json"""
    #item = json_try(item)
    if dump_method == 'yaml':
        return yaml_dump(item)
    elif dump_method == 'yaml_origin':
        return yaml_origin_dump(item)
    elif dump_method == 'json':
        #return json.dumps(item, ensure_ascii=False)
        return json_dump(item)
    else:
        raise NotImplementedError


def json_try(item):
    """iteratively try to convert all values to json dict"""
    if isinstance(item, str):
        try:
            x = json.loads(item)
            if not isinstance(x, str):
                return json_try(x)
            else:
                return x
        except json.JSONDecodeError:
            return item
    elif isinstance(item, dict):
        data = {}
        for key, value in item.items():
            data[key] = json_try(value)
        return data if len(data) > 0 else None
    elif isinstance(item, list):
        data = []
        for x in item:
            data.append(json_try(x))
        return data if len(data) > 0 else None
    else:
        return item


def my_load(string, dump_method):
    """my load method, yaml or json"""
    if dump_method == 'yaml':
        return yaml_load(string)
    elif dump_method == 'json':
        return json.loads(string)
    else:
        raise NotImplementedError


def my_input_format(
        messages: List[Dict],
        objects: List[Dict],
        choices: List[Dict],
        config: Union[str, Dict]
    ):
    """
    Process the input messages, global_arguments, tools, tool_choice,
        and convert it into a input string.
    The global arguments and tools can not be both empty.
    parameters:
        messages: List[Dict]
            the input messages
            For example:
        global_arguments: Dict
            Some global arguments you want the model to output.
            Global arguments can be regarded as COT of function call mode.
            For example:
        tools: List[Dict]
            the tools list you can use
            For example:
        tool_choice: Optional[Dict]
            choose a specific tool to use
            For example:
    """
    if isinstance(config, str):
        config = yaml_load(open(config, 'r', encoding='utf-8').read())
    if objects is not None and len(objects) > 0:
        string = my_dump(objects, config['input_dump_method'])
        objects_string = config['objects_template'].format(objects=string)
    else:
        objects_string = ""
    if choices is not None and len(choices) > 0:
        choice_names = ", ".join([choice['name'] if isinstance(choice, dict) else choice for choice in choices if choice is not None])
        choice_string = config['choice_template'].format(choices=choice_names)
    else:
        choice_string = ""
    system_suffix = config['system_template'].format(
        objects_string=objects_string.strip(),
        choice_string=choice_string.strip()
    ).strip()
    dialog = messages
    sys_msg_idx = find_system_msg(dialog)
    if sys_msg_idx == -1:
        dialog.insert(0, {"role": "system", "content": system_suffix})
    else:
        dialog[sys_msg_idx]["content"] += "\n" + system_suffix

    # merge functions
    #dialog = merge_messages(dialog)
    #dialog.append({"role": "system", "content": system_suffix})

    formated_messages = [message_format(msg, config) for msg in dialog]
    for x in formated_messages:
        if x is None:
            return None
    #input_string = "\n".join([message_format(msg, config).strip() for msg in dialog])
    input_string = "\n".join([fm.strip() for fm in formated_messages])
    input_string += config.get('starter', '')
    input_string = input_string.replace("json", "yaml")
    return input_string

def my_output_format(
        output: Dict,
        config: Union[str, Dict]
    ):
    if isinstance(config, str):
        config = yaml_load(open(config, 'r', encoding='utf-8').read())
    #return my_dump(output, config['output_dump_method'])
    if 'object_calls' in output:
        if config['output_dump_method'] == 'yaml':
            return yaml_dump_all(output['object_calls']).strip()
        elif config['output_dump_method'] == 'yaml_origin':
            return yaml_origin_dump_all(output['object_calls']).strip()
        elif config['output_dump_method'] == 'json':
            return json.dumps(output['object_calls'], ensure_ascii=False).strip()
        else:
            raise NotImplementedError
    else:
        return ''
