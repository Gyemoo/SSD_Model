import json


with open('/home/urp6/workspace/ssd_for_KAISTPD/official_Evaluation/kaist_annotations_test20.json','r') as j:
    annot_file = json.load(j)


anot = annot_file['annotations']

heightlist = []
small = 0
medium = 0
large = 0
for i in range(len(anot)):
    heightlist.append(anot[i]['height'])
    if anot[i]['height'] <= 55:
        small+= 1
    elif anot[i]['height'] <= 115:
        medium+= 1
    else:
        large += 1


import pdb;pdb.set_trace()
print()