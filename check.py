import json

with open('metrics.json', 'r') as file:
    data = json.load(file)

score = data['roc_auc_score']
print(score)
