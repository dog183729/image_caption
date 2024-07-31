import json
from collections import Counter

# Load the JSON data
with open('captions_val2017_with_tags.json', 'r') as f:
    val_data = json.load(f)
with open('captions_train2017_with_tags.json', 'r') as f:
    train_data = json.load(f)

# Combine tags from both datasets
all_tags_val = set(val_data['all_tags'])
all_tags_train = set(train_data['all_tags'])
all_tags_union = all_tags_val.union(all_tags_train)

# Manually check annotation and filter out invalid tags
invalid_tags = {'next', 'go', 'come', 'take'}
all_tags_filtered = {tag for tag in all_tags_union if tag not in invalid_tags}

# Count tag frequencies in both datasets
tag_counter = Counter()

for item in val_data['annotations']:
    for tag in item['tags']:
        if tag in all_tags_filtered:
            tag_counter[tag] += 1

for item in train_data['annotations']:
    for tag in item['tags']:
        if tag in all_tags_filtered:
            tag_counter[tag] += 1

# Filter out tags with frequency lower than 500
final_tags = {tag for tag, count in tag_counter.items() if count >= 500}

# Update the all_tags fields in both datasets
val_data['all_tags'] = list(final_tags)
train_data['all_tags'] = list(final_tags)

# Save the updated JSON data
with open('captions_val2017_with_tags_updated.json', 'w') as f:
    json.dump(val_data, f, indent=4)
with open('captions_train2017_with_tags_updated.json', 'w') as f:
    json.dump(train_data, f, indent=4)

print("All tags have been updated and saved to new JSON files.")
