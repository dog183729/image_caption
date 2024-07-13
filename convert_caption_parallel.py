import json
from multiprocessing import Process, Queue, cpu_count
import spacy
from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser

nlp = spacy.load("en_core_web_sm")

def load_data(q1, annotations):
    for item in annotations:
        q1.put(item)
    
    q1.put(None)  # Signal to stop

def parse_caption(q1, q2, device):
    parser = SceneGraphParser('lizhuang144/flan-t5-base-VG-factual-sg', device=device)    
    while True:
        item = q1.get()
        if item is None:
            q2.put(None)
            break
        caption = item.get('caption', '')
        graph_obj = parser.parse([caption], beam_size=1, return_text=False, max_output_len=128)
        item['graph_obj'] = graph_obj[0]
        q2.put(item)

def tag_filtering(q2, output_queue):
    import spacy
    nlp = spacy.load("en_core_web_sm")
    
    while True:
        item = q2.get()
        if item is None:
            output_queue.put(None)
            break
        graph_obj = item['graph_obj']
        entities = graph_obj['entities']
        relations = graph_obj['relations']
        
        tags = set()
        for entity in entities:
            entity_text = entity['head']
            attributes = entity['attributes']
            tags.add(entity_text)
            tags.update(attributes)
        
        for relation in relations:
            tags.add(relation['subject'])
            tags.add(relation['relation'])
            tags.add(relation['object'])
        
        doc = nlp(" ".join([str(tag) for tag in tags]))
        filtered_tags = set()
        for token in doc:
            if token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                filtered_tags.add(token.lemma_)
        
        item['tags'] = list(filtered_tags)
        output_queue.put(item)

def main():
    with open('captions_val2017.json', 'r') as f:
        data = json.load(f)

    annotations = data['annotations']
    input_queue = Queue()
    parsed_queue = Queue()
    output_queue = Queue()

    num_workers = 1

    import torch
    device = "cpu"

    loader = Process(target=load_data, args=(input_queue, annotations))
    parser_processes = [Process(target=parse_caption, args=(input_queue, parsed_queue, device)) for _ in range(num_workers)]
    filter_processes = [Process(target=tag_filtering, args=(parsed_queue, output_queue)) for _ in range(num_workers)]

    loader.start()
    for p in parser_processes:
        p.start()
    for f in filter_processes:
        f.start()

    results = []
    all_tags_set = set()
    finished_workers = 0

    while True:
        result = output_queue.get()
        if result is None:
            finished_workers += 1
            if finished_workers == num_workers:
                break
        else:
            results.append(result)
            all_tags_set.update(result['tags'])

    for p in parser_processes + filter_processes:
        p.join()
    loader.join()

    data['annotations'] = results
    data['all_tags'] = list(all_tags_set)

    with open('captions_val2017_with_tags.json', 'w') as f:
        json.dump(data, f, indent=4)

    print("Processed data has been saved to captions_val2017_with_tags.json")

if __name__ == "__main__":
    main()
