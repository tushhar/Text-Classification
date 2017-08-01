import csv
import json

print("reading document_set")
reader = csv.reader(open('document_set.csv', 'r'))
document_set = {}
next(reader)
for row in reader:
   k, v = row
   document_set[k] = v


print("reading training set now")
reader1 = csv.reader(open('Training_Data.csv', 'r'))
training_set = {}
next(reader1)
for row in reader1:
   k, v = row
   training_set[document_set.get(k)] = v

print("done ...  saving now")


with open('my_train_dict.json', 'w') as f:
    json.dump(training_set, f)