from tqdm import tqdm
import networkx as nx
import stanza
from nltk.tokenize import sent_tokenize
from copy import deepcopy as cp
import fitz
import io
import os

#тут реализована функция построения графа на основе текста. Построенный граф сливается с графом знаний

class GraphBuilder:
    def __init__(self):
        self.parser = stanza.Pipeline(lang='ru', processors='tokenize,pos,lemma,ner,depparse')
        self.normalizer = None
        self.work_dir = None

    def __scan_work_dir(self):
        if self.work_dir is None:
            raise "No work dir specified"
        
        fcnt = 0
        tcnt = 0
        pcnt = 0
        for fname in os.listdir(self.work_dir):
            if fname[-4:] == ".txt":
                if fname[:-4] in self.work_dir_index:
                    self.work_dir_index[fname[:-4]] = fname
                tcnt += 1                
            if fname[-4:] == ".pdf":
                self.work_dir_index[fname] = None
                pcnt += 1
            fcnt += 1

        print("======================")
        print(f"in \t{self.work_dir}")
        print(f"files\t {fcnt}")
        print(f"txts\t {tcnt}")
        print(f"pdfs\t {pcnt}")
        print("======================")

    def init_parser(self):
        if self.parser is None:
            self.parser = stanza.Pipeline(lang='ru', processors='tokenize,pos,lemma,ner,depparse')

    def init_normalizer(self):
        if self.normalizer is None:
            self.normalizer = stanza.Pipeline(lang='ru', processors='tokenize,lemma')

    def select_work_dir(self, dir_path):
        self.work_dir = dir_path
        self.work_dir_index = {} # это словарь соответствий pdf и txt файлов
        self.__scan_work_dir()

    def extract_text_from_pdfs(self):
        # для каждого .pdf файла в work_dir создает соответствующий ему .txt файл, содержащий извлеченный из него текстовый слой
        if self.work_dir is None:
            raise "No work dir specified"
        ccnt = 0
        for fname in os.listdir(self.work_dir):
            if fname[-4:] == ".pdf" and self.work_dir_index[fname] == None:
                self.work_dir_index[fname] = self.work_dir + "/" + fname + ".txt"
                ccnt += 1
                with fitz.open(self.work_dir + "/" + fname) as doc:  # open document
                    text = chr(12).join([page.get_text() for page in doc])
                with io.open(self.work_dir + "/" + fname + ".txt", "w+", encoding='utf-8') as fout:
                    fout.write(text)      
        
        print(f"converted {ccnt} files")
        self.__scan_work_dir()

    def add_all_files_to_graph(self, graph_structure, display_mode = "progress_bar"):
        # парсит и добавляет к графу все текстовые (.txt) файлы из self.work_dir
        for fname in tqdm(os.listdir(self.work_dir)) if display_mode == "progress_bar" else os.listdir(self.work_dir):
            if fname[-4:] == ".txt":
                if display_mode == "progress_bar":
                    graph_structure = self.parse_file_and_fill_graph(graph_structure, self.work_dir + "/" + fname, trace = False)
                else:
                    print(f"loading {fname}")
                    graph_structure = self.parse_file_and_fill_graph(graph_structure, self.work_dir + "/" + fname, trace = True)
        return graph_structure        

    def apply_node_frequency_limit(self, graph_structure, limit = 2):
        cnt = 0
        node_list = cp(graph_structure.G.nodes())
        for node in node_list:
            if len(node_list[node]["locations"]) < limit:
                graph_structure.remove_node(node)
                cnt += 1
        print(f"removed {cnt} nodes")
        return graph_structure

    def apply_edge_frequency_limit(self, graph_structure, limit = 2):
        cnt = 0
        edge_list = cp(graph_structure.G.edges())
        for node in edge_list:
            if len(edge_list[node]["locations"]) < limit:
                graph_structure.remove_edge(node)
                cnt += 1
        print(f"removed {cnt} edges")
        return graph_structure
    
    def apply_word_reasonability_check(self, graph_structure):
        cnt = 0
        russian_words = {}
        with io.open("ru", encoding = "utf-8") as f:
            for word in f.readlines():
                russian_words[word[:-1]] = ''
        node_list = cp(graph_structure.G.nodes())
        for node in node_list:
            if not node in russian_words:
                graph_structure.remove_node(node)
                cnt += 1
        print(f"removed {cnt} nodes")
        return graph_structure        

    def parse_file_and_fill_graph(self, graph_structure, fname, trace = True):
        text = ''
        with io.open(fname , encoding='utf-8') as inp:
            text += inp.read()
        # текст загружен, теперь надо сохранить инфу о документе
        doc_name = os.path.splitext(fname)[0]
        graph_structure.document_base[doc_name] = {'fname' : fname, 'len' : len(text)}
        # биение текста на предложения
        current_document = [sent for sent in sent_tokenize(text, language="russian")]
        # если нет парера, инициализируем парсер
        self.init_parser()
        # делаем новый граф. в него будем записывать сущности и отношения, извлеченные из текста
        current_document_graph = nx.DiGraph() # граф, который будет строиться во время обработки документа, а затем сольется с исходным графом
        # проход по предложениям в документе
        for s in tqdm(current_document) if trace else current_document:
            sent_repr = self.parser(s).sentences[0]
            parse_d = {}
            # преобразование вывода пайплайна в словарь, содержащий только необходимые ключи
            for word in sent_repr.words:
                parse_d[word.text] = {"head": sent_repr.words[word.head-1].text, 
                                    "dep": word.deprel, 
                                    "id": word.id, 
                                    "upos": word.upos,
                                    "lem": word.lemma.lower(),
                                    "start" : word.start_char,
                                    "end" : word.end_char}
            # правила добавления вершин и ребер к графу
            pair = [None, None]
            con_anot = None
            for elm in parse_d.keys():
                # определяем, делаем ли мы из этого вершины и ребра
                if     parse_d[elm]['dep'] == 'amod'\
                    or parse_d[elm]['dep'] == 'nmod'\
                    or parse_d[elm]['dep'] == 'flat:name'\
                    or parse_d[elm]['dep'] == 'appos'\
                    or parse_d[elm]['dep'] == 'advmod'\
                    or parse_d[elm]['dep'] == 'nsubj':
                    # обработка пар "объект - свойство"
                    # amod, nmod - свойства
                    # appos - пояснительное дополнение, 
                    # для nsubj может есть смысл силой приводить связь к prp если второе слово - adj
                    pair[0] = {'pos' : parse_d[elm]['upos'], 
                                'lem' : parse_d[elm]['lem'],
                                'loc' : (parse_d[elm]['end'] +\
                                        parse_d[elm]['start']) / 2}
                    pair[1] = {'pos' : parse_d[parse_d[elm]['head']]['upos'], 
                                'lem' : parse_d[parse_d[elm]['head']]['lem'],
                                'loc' : (parse_d[parse_d[elm]['head']]['end'] +\
                                        parse_d[parse_d[elm]['head']]['start']) / 2}
                else:              
                    if      parse_d[elm]['dep'] == 'conj'\
                        and parse_d[parse_d[elm]['head']]['dep'] != 'root': 
                        # добавление свойств, связвнных с объектом через перечисление
                        # объекты или свойства, соединенные через союз или пунктуацию
                        # не включаем объекты
                        if parse_d[elm]['upos'] not in ['ADJ', 'ADV']:
                            continue
                        pair[0] = {'pos' : parse_d[elm]['upos'], 
                                    'lem' : parse_d[elm]['lem'],
                                    'loc' : (parse_d[elm]['end'] +\
                                            parse_d[elm]['start']) / 2}
                        # добавление второго элемента пары
                        # проходим по дереву на два шага назад, вместо одного
                        pair[1] = {'pos' : parse_d[parse_d[parse_d[elm]['head']]['head']]['upos'], 
                                    'lem' : parse_d[parse_d[parse_d[elm]['head']]['head']]['lem'],
                                    'loc' : (parse_d[parse_d[parse_d[elm]['head']]['head']]['end'] +\
                                            parse_d[parse_d[parse_d[elm]['head']]['head']]['start']) / 2}
                    else:
                        continue
                        
                # не записываем определенные части речи
                if pair[0]['pos'] in ['PART', 'CCONJ', 'PUNC', 'SCONJ', 'ADP', 'NUM', 'DET', 'PRON', 'X']:
                    continue
                if pair[1]['pos'] in ['PART', 'CCONJ', 'PUNC', 'SCONJ', 'ADP', 'NUM', 'DET', 'PRON', 'X']:
                    continue
                    
                # добавление вершин
                current_document_graph.add_node(pair[0]['lem'])
                current_document_graph.add_node(pair[1]['lem'])
                # Добавление ребер
                cur_edge = (pair[0]['lem'], pair[1]['lem'])
                current_document_graph.add_edge(*cur_edge)
                
                # добавление свойств вершин и ребер
                con_anot = ''
                if parse_d[elm]['dep'] == 'appos':
                    con_anot = 'syn'
                if parse_d[elm]['dep'] == 'amod':
                    con_anot = 'prp'
                if parse_d[elm]['dep'] == 'nmod':
                    con_anot = 'clr'
                if parse_d[elm]['dep'] == 'conj':
                    con_anot = 'conj prp'
                if parse_d[elm]['dep'] == 'advmod':
                    con_anot = 'act prp'
                if parse_d[elm]['dep'] == 'nsubj': 
                    if pair[1]['pos'] in ['VERB', 'AUX']:
                        con_anot = 'act'
                    if pair[1]['pos'] in ['ADJ', 'ADV']:
                        con_anot = 'prp'
                if parse_d[elm]['dep'] == 'flat:name':
                    con_anot = 'name pt'
                
                # добавление аттрибутов узлов и ребер
                # название ребра
                current_document_graph.edges[cur_edge].setdefault("label", [0, (con_anot + ' ') if con_anot != None else ''])
                current_document_graph.edges[cur_edge]["label"][0] = current_document_graph.edges[cur_edge]["label"][0] + 1
                # позиция, в которой встречена связь
                current_document_graph.edges[cur_edge].setdefault('locations', [])
                current_document_graph.edges[cur_edge]['locations'] =\
                    current_document_graph.edges[cur_edge]['locations'] +\
                    [{'t' : doc_name, 'p' : (pair[0]['loc'] + pair[1]['loc']) / 2}]
                # позиции, в которых встречены узлы
                current_document_graph.nodes[pair[0]['lem']].setdefault('locations', [])
                current_document_graph.nodes[pair[0]['lem']]['locations'] =\
                    current_document_graph.nodes[pair[0]['lem']]['locations'] +\
                    [{'t' : doc_name, 'p' : pair[0]['loc']}]
                    
                current_document_graph.nodes[pair[1]['lem']].setdefault('locations', [])
                current_document_graph.nodes[pair[1]['lem']]['locations'] =\
                    current_document_graph.nodes[pair[1]['lem']]['locations'] +\
                    [{'t' : doc_name, 'p' : pair[1]['loc']}]
                
                # добавление цветов
                for element in pair:
                    if element['pos'] == 'NOUN': # существительные (объекты)
                        current_document_graph.nodes[element['lem']]["color"] = 'pink'
                    else:
                        if element['pos'] in ['ADJ', 'ADV']: # прилагательные (свойства)
                            current_document_graph.nodes[element['lem']]["color"] = 'orange'
                        else:
                            if element['pos'] == 'PROPN': # имена собственные
                                current_document_graph.nodes[element['lem']]["color"]  = 'red'
                            else:
                                if element['pos'] in ['VERB', 'AUX']: # глаголы
                                    current_document_graph.nodes[element['lem']]["color"]  = 'cyan'
                                else:
                                    current_document_graph.nodes[element['lem']]["color"]  = 'yellow'   

        # добавляем построенный граф в структурку графа знаний
        graph_structure.__merge_graph__(current_document_graph)
        return graph_structure
    

    def create_syntax_graph_for_sentence(self, text):
        self.init_parser()

        sentence_graph = nx.DiGraph()
        sent_repr = self.parser(text).sentences[0]
        parse_d = {}

        for word in sent_repr.words:
            parse_d[word.text] = {"head": sent_repr.words[word.head-1].text, 
                                  "dep": word.deprel,
                                  "upos": word.upos,
                                  "lem": word.lemma.lower(),
                                  "start" : word.start_char,
                                  "end" : word.end_char}
            pair = [None, None]

        for i, elm in enumerate(parse_d.items()):
            if i % 3 == 0:
                print("".join(["-" for _ in range(80)]))
            print(f"{elm[1]['lem']}",end="")
            print("".join([" " for _ in range(20 - len(elm[1]['lem']))]), end = "")
            print(f"{elm[1]['dep']}",end="")
            print("".join([" " for _ in range(20 - len(elm[1]['dep']))]), end = "")
            print(f"{elm[1]['upos']}",end="")
            print("".join([" " for _ in range(20 - len(elm[1]['upos']))]), end = "")
            print(f"{elm[1]['head']}")
        print("".join(["-" for _ in range(80)]))

        for elm in parse_d.keys():
            is_sent_core = False

            # определяем, делаем ли мы из этого вершины и ребра
            if     parse_d[elm]['dep'] != 'root':

                if parse_d[parse_d[elm]['head']]['dep'] == 'root':
                    is_sent_core = True
                    
                # обработка пар "объект - свойство"
                # amod, nmod - свойства
                # appos - пояснительное дополнение, 
                # для nsubj может есть смысл силой приводить связь к prp если второе слово - adj
                pair[0] = {'pos' : parse_d[elm]['upos'], 
                            'lem' : parse_d[elm]['lem'] + " (" + str(parse_d[elm]['start']) + ")",
                            'loc' : (parse_d[elm]['end'] +\
                                    parse_d[elm]['start']) / 2}
                pair[1] = {'pos' : parse_d[parse_d[elm]['head']]['upos'], 
                            'lem' : parse_d[parse_d[elm]['head']]['lem'] + " (" + str(parse_d[parse_d[elm]['head']]['start']) + ")", 
                            'loc' : (parse_d[parse_d[elm]['head']]['end'] +\
                                    parse_d[parse_d[elm]['head']]['start']) / 2}
            else:              
                if      parse_d[elm]['dep'] == 'conj'\
                    and parse_d[parse_d[elm]['head']]['dep'] != 'root': 
                    # добавление свойств, связвнных с объектом через перечисление
                    # объекты или свойства, соединенные через союз или пунктуацию
                    # не включаем объекты
                    if parse_d[elm]['upos'] not in ['ADJ', 'ADV']:
                        continue

                    pair[0] = {'pos' : parse_d[elm]['upos'], 
                                'lem' : parse_d[elm]['lem'] + " (" + str(parse_d[elm]['start']) + ")",
                                'loc' : (parse_d[elm]['end'] +\
                                        parse_d[elm]['start']) / 2}
                    # добавление второго элемента пары
                    # проходим по дереву на два шага назад, вместо одного
                    pair[1] = {'pos' : parse_d[parse_d[parse_d[elm]['head']]['head']]['upos'], 
                                'lem' : parse_d[parse_d[parse_d[elm]['head']]['head']]['lem'] + " (" + str(parse_d[parse_d[parse_d[elm]['head']]['start']]) + ")",
                                'loc' : (parse_d[parse_d[parse_d[elm]['head']]['head']]['end'] +\
                                        parse_d[parse_d[parse_d[elm]['head']]['head']]['start']) / 2}
                else:
                    continue
                    
            # не записываем определенные части речи
            if pair[0]['pos'] in ['PART', 'CCONJ', 'PUNC', 'PUNCT', 'SCONJ', 'ADP', 'NUM', 'DET', 'X']:
                continue
            if pair[1]['pos'] in ['PART', 'CCONJ', 'PUNC', 'PUNCT', 'SCONJ', 'ADP', 'NUM', 'DET', 'X']:
                continue
                
            print(pair)
            # добавление вершин
            sentence_graph.add_node(pair[0]['lem'])
            sentence_graph.add_node(pair[1]['lem'])
            # Добавление ребер
            cur_edge = (pair[0]['lem'], pair[1]['lem'])
            print(cur_edge)
            sentence_graph.add_edge(*cur_edge)
                
            con_anot = ""
            doc_name = ""
            # добавление аттрибутов узлов и ребер
            # название ребра
            sentence_graph.edges[cur_edge].setdefault("label", [0, (con_anot + ' ') if con_anot != None else ''])
            sentence_graph.edges[cur_edge]["label"][0] = sentence_graph.edges[cur_edge]["label"][0] + 1
            # название ребра
            sentence_graph.edges[cur_edge].setdefault("dep", parse_d[elm]['dep'])
            # позиция, в которой встречена связь
            sentence_graph.edges[cur_edge].setdefault('locations', [])
            sentence_graph.edges[cur_edge]['locations'] =\
                sentence_graph.edges[cur_edge]['locations'] +\
                [{'t' : doc_name, 'p' : (pair[0]['loc'] + pair[1]['loc']) / 2}]
            # позиции, в которых встречены узлы
            sentence_graph.nodes[pair[0]['lem']].setdefault('locations', [])
            sentence_graph.nodes[pair[0]['lem']]['locations'] =\
                sentence_graph.nodes[pair[0]['lem']]['locations'] +\
                [{'t' : doc_name, 'p' : pair[0]['loc']}]
                
            sentence_graph.nodes[pair[1]['lem']].setdefault('locations', [])
            sentence_graph.nodes[pair[1]['lem']]['locations'] =\
                sentence_graph.nodes[pair[1]['lem']]['locations'] +\
                [{'t' : doc_name, 'p' : pair[1]['loc']}]
            # =============
            sentence_graph.nodes[pair[0]['lem']].setdefault('pos', pair[0]['pos'])
                
            sentence_graph.nodes[pair[1]['lem']].setdefault('pos', pair[1]['pos'])
                        
            # добавление цветов
            for i, element in enumerate(pair):
                
                if is_sent_core:
                    sentence_graph.nodes[element['lem']]["color"] = 'black'
                    continue
                
                try: # если вершина уже раскрашена, то не перекрашиваем
                    _ = sentence_graph.nodes[element['lem']]["color"]
                    continue
                except KeyError:
                    pass

                sentence_graph.nodes[element['lem']]["color"]  = 'white'
        return sentence_graph
    
    def extract_noun_phrases(self, text, clear_hanging_nodes = False, add_focus = False, trace = False):
        # вернуть только имена существительные из именных групп
        self.init_parser()

        noun_graph = nx.Graph()

        sent_repr = self.parser(text).sentences[0]
        parse_d = {}

        for word in sent_repr.words:
            parse_d[word.text] = {"head": sent_repr.words[word.head-1].text, 
                                  "dep": word.deprel,
                                  "upos": word.upos,
                                  "lem": word.lemma.lower()}
            
        if trace:
            for i, elm in enumerate(parse_d.items()):
                if i % 3 == 0:
                    print("".join(["-" for _ in range(80)]))
                print(f"{elm[1]['lem']}",end="")
                print("".join([" " for _ in range(20 - len(elm[1]['lem']))]), end = "")
                print(f"{elm[1]['dep']}",end="")
                print("".join([" " for _ in range(20 - len(elm[1]['dep']))]), end = "")
                print(f"{elm[1]['upos']}",end="")
                print("".join([" " for _ in range(20 - len(elm[1]['upos']))]), end = "")
                print(f"{elm[1]['head']}")
            print("".join(["-" for _ in range(80)]))

        for elm in parse_d.keys():
            
            if parse_d[elm]['upos'] == 'NOUN':
                noun_graph.add_node(parse_d[elm]['lem'])
                #===========================================
                noun_graph.nodes[parse_d[elm]['lem']]["color"] = "red"
                noun_graph.nodes[parse_d[elm]['lem']]["locations"] = [1]
                #===========================================

                if parse_d[parse_d[elm]['head']]['upos'] == 'NOUN':
                    #===========================================
                    noun_graph.add_node(parse_d[parse_d[elm]['head']]['lem'])
                    #===========================================
                    noun_graph.nodes[parse_d[parse_d[elm]['head']]['lem']]["color"] = "red"
                    noun_graph.nodes[parse_d[parse_d[elm]['head']]['lem']]["locations"] = [1]
                    #===========================================
                    noun_graph.add_edge(parse_d[elm]['lem'], 
                                        parse_d[parse_d[elm]['head']]['lem'])
                    #===========================================
                    noun_graph.edges[parse_d[elm]['lem'], parse_d[parse_d[elm]['head']]['lem']]["label"] = "sub link"
                    noun_graph.edges[parse_d[elm]['lem'], parse_d[parse_d[elm]['head']]['lem']]["locations"] = [1]
                    #===========================================
                    
                if parse_d[parse_d[elm]['head']]['upos'] == 'VERB':
                    noun_graph.add_node(parse_d[parse_d[elm]['head']]['lem'])
                    #===========================================
                    noun_graph.nodes[parse_d[parse_d[elm]['head']]['lem']]["color"] = "blue"
                    noun_graph.nodes[parse_d[parse_d[elm]['head']]['lem']]["locations"] = [1]
                    #===========================================

                    noun_graph.add_edge(parse_d[elm]['lem'], 
                                        parse_d[parse_d[elm]['head']]['lem'])
                    #===========================================
                    noun_graph.edges[parse_d[elm]['lem'], parse_d[parse_d[elm]['head']]['lem']]["label"] = ""
                    noun_graph.edges[parse_d[elm]['lem'], parse_d[parse_d[elm]['head']]['lem']]["locations"] = [1]
                    #===========================================
#            else:
#                if parse_d[parse_d[elm]['head']]['upos'] == 'NOUN':
#                    noun_graph.add_node(parse_d[elm]['lem'])
#                    #===========================================
#                    noun_graph.nodes[parse_d[elm]['lem']]["color"] = "orange"
#                    noun_graph.nodes[parse_d[elm]['lem']]["locations"] = [1]
#                    #===========================================
#                    noun_graph.add_node(parse_d[parse_d[elm]['head']]['lem'])
#                    #===========================================
#                    noun_graph.nodes[parse_d[parse_d[elm]['head']]['lem']]["color"] = "red"
#                    noun_graph.nodes[parse_d[parse_d[elm]['head']]['lem']]["locations"] = [1]
#                    #===========================================
#                    noun_graph.add_edge(parse_d[elm]['lem'], 
#                                        parse_d[parse_d[elm]['head']]['lem'])
#                    #===========================================
#                    noun_graph.edges[parse_d[elm]['lem'], parse_d[parse_d[elm]['head']]['lem']]["label"] = "p link"
#                    noun_graph.edges[parse_d[elm]['lem'], parse_d[parse_d[elm]['head']]['lem']]["locations"] = [1]
#                    #===========================================
#						

        for node in cp(noun_graph.nodes):
            # заменяем глаголы на "глагольные ссылки"
            if noun_graph.nodes[node]['color'] == "blue":
                for neighbor_node_1 in noun_graph.neighbors(node):
                    for neighbor_node_2 in noun_graph.neighbors(node):
                        if neighbor_node_1 != neighbor_node_2:
                            noun_graph.add_edge(neighbor_node_1, neighbor_node_2)
                            #===========================================
                            noun_graph.edges[neighbor_node_1, neighbor_node_2]["label"] = "verb link"
                            noun_graph.edges[neighbor_node_1, neighbor_node_2]["locations"] = [1]
                            #===========================================
                noun_graph.remove_node(node)
            
        if clear_hanging_nodes:
            for node in cp(noun_graph.nodes):
                # удаляем висячие узлы
                if len([_ for _ in noun_graph.neighbors(node)]) == 0:
                    noun_graph.remove_node(node)

        if add_focus: # в add focus лежат слова из предложения, которые надо добавить как фокус-узлы


            for focus_element in add_focus:
                
                focus_element_repr = self.parser(focus_element).sentences[0]
                parse_d = {}

                for word in focus_element_repr.words:
                    parse_d[word.text] = {"head": sent_repr.words[word.head-1].text, 
                                        "dep": word.deprel,
                                        "upos": word.upos,
                                        "lem": word.lemma.lower()}
                    
                for word in parse_d:
                    node_norm = parse_d[word]["lem"]
                        
                    noun_graph.add_node(node_norm)
                    #===========================================
                    noun_graph.nodes[node_norm]["color"] = "purple"
                    noun_graph.nodes[node_norm]["locations"] = [1]
                    #===========================================


        return noun_graph


    def calculate_df(self, document): # document это список предложений 
        self.init_normalizer()
        df = {}

        for sentence in document:
            sent_repr = self.normalizer(sentence).sentences[0]
            # преобразование вывода пайплайна в словарь, содержащий только необходимые ключи
            for word in sent_repr.words:
                norm = word.lemma.lower()
                df.setdefault(norm, 0)
                df[norm] += 1
        
        return df
    
    def calculat_tf(self, sentence):
        self.init_normalizer()
        tf = {}

        sent_repr = self.normalizer(sentence).sentences[0]
        # преобразование вывода пайплайна в словарь, содержащий только необходимые ключи
        for word in sent_repr.words:
            norm = word.lemma.lower()
            tf.setdefault(norm, 0)
            tf[norm] += 1
        
        return tf
    
    def find_most_significant_words(self, sentence, top_p, df, trace = False):
        tf = self.calculat_tf(sentence)
        tfidf = {}
        for word in tf.keys():
            tfidf[word] = tf[word] / df[word]
            if tfidf[word] == 1:
                tfidf[word] = 0 # нет смысла брать слова, которые только в этом предложении встретились, потому что они бесполезны бли линкинга
        top_p_words_num = int(top_p * len(tf.keys()))
        if trace:
            print(tfidf)
        top_p_words = dict(list(sorted(tfidf.items(), key = lambda x: x[1], reverse = True))[:top_p_words_num])
        return top_p_words


    def filter_graph_(self, graph, focus_nodes_lemmatized = [], core_radius = 1, focus_radius = 1):
        # RADIUSES 0 OR 1
        filterer_graph = nx.Graph(graph)
        filtered_graph = nx.Graph(graph)
        for node in filterer_graph.nodes:
            hold = False
            if filterer_graph.nodes[node]['color'] == 'black':
                hold = True
            if node in focus_nodes_lemmatized:
                hold = True
            
            for neighbor_node in filterer_graph.neighbors(node):
                if filterer_graph.nodes[neighbor_node]['color'] == 'black' and core_radius == 1:
                    hold = True
                    break
                if node in focus_nodes_lemmatized and focus_radius == 1:
                    hold = True
                    break
            if not hold:
                filtered_graph.remove_node(node)
        return filtered_graph

    # задача
    # tf-idf находит слова, которые характеры для данного предложения (берем тип топ 10-20% слов)
    # эти слова идут в focus_nodes_lemmatized
    # полученные после фильтрации графы идут в алгоритм слияния графов 
