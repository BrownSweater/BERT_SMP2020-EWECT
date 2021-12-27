import json
from tqdm import trange
from harvesttext import HarvestText
import pyhanlp
import re
import os

def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print('%s -> data over' % file)
    return data

def save_json(data, file, indent=1):
    with open(file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, indent=1, ensure_ascii=False))
    print('data -> %s over' % file)

def remove_url(src):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', src, flags=re.MULTILINE)
    return vTEXT

# file: 数据文件
def clean_text(file, save_dir):
    ht = HarvestText()
    CharTable = pyhanlp.JClass('com.hankcs.hanlp.dictionary.other.CharTable')
    data = read_json(file)
    num_null = 0
    cleaned_data = []
    for i in trange(len(data)):
        content = CharTable.convert(data[i]['content'])
        cleaned_content = remove_url(ht.clean_text(content, emoji=False))  # 过滤@后最多6个字符
        num_null += 1 if cleaned_content == '' else 0
        if 'train' in file and (not content or not cleaned_content):  # 删除train中的自带的空数据或清洗后出现的空数据
            continue
        if 'eval' in file or 'test' in file:
            cleaned_data.append({'id':data[i]['id'], 'content':cleaned_content, 'label':data[i]['label']})
        else:
            cleaned_data.append({'id':data[i]['id'], 'content':cleaned_content, 'label':data[i]['label']})
    filename = file.split('/')[-1]
    save_json(cleaned_data, os.path.join(save_dir, filename))
    print('num data: ', num_null)

clean_text('./data/raw/virus_train.txt', './data/clean')
clean_text('./data/raw/usual_train.txt', './data/clean')


clean_text('./data/raw/virus_test_labeled.txt', './data/clean')
clean_text('./data/raw/usual_test_labeled.txt', './data/clean')


clean_text('./data/raw/virus_eval_labeled.txt', './data/clean')
clean_text('./data/raw/usual_eval_labeled.txt', './data/clean')