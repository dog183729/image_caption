import json
import spacy
from factual_scene_graph.parser.scene_graph_parser import SceneGraphParser
import torch

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Initialize SceneGraphParser
parser = SceneGraphParser('lizhuang144/flan-t5-base-VG-factual-sg', device=device)

def process_caption_with_sgp(caption):
    # Use SceneGraphParser to extract objects, attributes, and relations
    graph_obj = parser.parse([caption], beam_size=1, return_text=False, max_output_len=128)
    
    entities = graph_obj[0]['entities']
    relations = graph_obj[0]['relations']
    
    # Extract nouns, verbs, and adjectives from objects, attributes, and relations
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
    
    # Use spaCy for POS tagging and lemmatization
    doc = nlp(" ".join([str(tag) for tag in tags]))  # Ensure all tags are strings
    filtered_tags = set()
    for token in doc:
        if token.pos_ in ['NOUN', 'VERB', 'ADJ']:
            filtered_tags.add(token.lemma_)
    
    return list(filtered_tags)

# Read JSON data
with open('captions_val2017.json', 'r') as f:
    data = json.load(f)

annotations = data['annotations']
all_tags_set = set()

# Process each caption and add new fields
for item in annotations:
    caption = item.get('caption', '')
    processed_tags = process_caption_with_sgp(caption)
    item['tags'] = processed_tags
    all_tags_set.update(processed_tags)

# Add all tags to the data
data['all_tags'] = list(all_tags_set)

# Save processed data to a new JSON file
with open('captions_val2017_with_tags.json', 'w') as f:
    json.dump(data, f, indent=4)

print("Processed data has been saved to captions_val2017_with_tags.json")
