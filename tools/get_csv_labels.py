import glob
import os
import csv
import sys

classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
path = sys.argv[1]

with open(path + '.csv', 'w', newline='') as csvfile:
    fieldnames = classes+['image']
    csvwriter = csv.DictWriter(csvfile, delimiter=',',
                               quoting=csv.QUOTE_NONE,
                               fieldnames=fieldnames)
    headers = dict( (n,n) for n in fieldnames )
    csvwriter.writerow(headers)
    for class_name in classes:
        images = glob.glob(os.path.join(path, class_name, '*.jpg'))
        class_idx = classes.index(class_name)
        for img in images:
            data = {}
            for i in range(len(classes)):
                data[classes[i]] = float(i == class_idx)
            data['image'] = os.path.basename(img)
            csvwriter.writerow(data)
