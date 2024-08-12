import math

def remap_keys(dict):
    return [{'key' : k, 'value' : v} for k, v in dict.items()]

def remap_back(mapping):
    return {tuple(d['key']) : d['value'] for d in mapping}

def shanon_entropy(parse_d):
    subtree_str = ''
    for elm in parse_d:
        subtree_str += ' ' + elm
    str_elements = set(subtree_str)
    entropy = 0
    for elm in str_elements:
        prob = subtree_str.count(elm) / len(subtree_str)
        entropy -= prob * math.log2(prob)
        
    return entropy

def intersection(lhs, rhs):
    # пересечение ключей словарей
    intersection_ = []
    for elm in lhs.keys():
        try:
            _ = rhs[elm]
            intersection_.append(elm)
        except KeyError:
            pass
    return intersection_

def union(lhs, rhs):
    # пересечение ключей словарей
    union_ = {}
    for elm in lhs.keys():
        union_[elm] = " "
    for elm in rhs.keys():
        union_[elm] = " "
        
    return list(union_.keys())

def list_intersection(lhs, rhs):
    # пересечение ключей словарей
    intersection_ = []
    for elm in lhs:
        if elm in rhs:
            intersection_.append(elm)
    return intersection_

def list_union(lhs, rhs):
    # пересечение ключей словарей
    union_ = []
    for elm in lhs:
        if not elm in union_:            
            union_.append(elm)
            
    return union_

def sigmoid(x, c = 1):
    return 1 / (1 + math.exp(-c * x))