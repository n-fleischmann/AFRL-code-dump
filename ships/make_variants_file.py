import os
import xml.etree.ElementTree as ET

VOC_path = '/data/ships/nate/ShipRSImageNet/VOC_Format'
anno_folder = os.path.join(VOC_path, 'Annotations')

all_variants = {}

for file in os.listdir(anno_folder):
    path = os.path.join(anno_folder, file)
    root = ET.parse(path).getroot()

    for child in root:
        if child.tag == 'object':
            name = child.find('name').text
            level_0 = int(child.find('level_0').text)
            level_2 = int(child.find('level_2').text)
            if level_0 != 2:
                all_variants[name] = level_2

list_variants = [(k,v ) for k, v in all_variants.items()]
list_variants = sorted(list_variants, key=lambda x: x[1])

# print(list_variants)
with open(os.path.join(VOC_path, 'variants.txt'), 'w') as outfile:
    for variant in list_variants:
        # print(f"{variant[0]} {variant[1]}\n")
        outfile.write(f"{' '.join(variant)}\n")

# print("="*30)
# for k, v in all_variants.items(): print(k, v)
# print("="*30)
# classes = all_variants.values()
# max_val = max(classes)
# for n in range(1, max_val+1):
#     if n not in classes: print(n)