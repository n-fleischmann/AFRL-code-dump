import os
import re

# This is where the splits are definted
# dict[2nd level class id: OOD-Group]


# Easy Splits -> ~5.8k Examples of Level 1 OOD
#                ~1.1k Examples of Level 2 OOD
#                ~6.1k Examples ID
military_split = {
    1: 1,  # "Other ship" categories are always OOD
    2: 2,  # Other Military
    12: 1, # Other merchant
    13: 1, # Container Ship
    14: 1, # RoRo
    15: 1, # Cargo
    16: 1, # Barge
    17: 1, # Tugboat
    18: 1, # Ferry
    19: 1, # Yacht
    20: 1, # Sailboat
    21: 1, # Fishing Vessel
    22: 1, # Oil Tanker
    23: 1, # Hovercraft
    24: 1  # Motorboat
}

civ_split = {
    1: 1,  # Other Ship
    2: 1,  # Other Military
    12: 2, # Other merchant
    13: 2, # Container Ship
    20: 2, # Sailboat
    3: 1,  # Submarine
    4: 1,  # Aircraft Carrier
    5: 1,  # Ticonderoga
    6: 1,  # Destroyer
    7: 1,  # Frigate
    8: 1,  # Patrol
    9: 1,  # Landing
    10: 1, # Commander
    11: 1, # Aux
}



# Hard Splits -> ~1.3K Examples of Level 1 OOD
#                ~4.9k Examples of Level 2 OOD
#                ~6.8k Examples ID
hard1_split = {
    1: 1,  # Other Ship
    # Military
    2: 2,  # Other Military
    5: 2,  # Ticonderoga
    8: 2,  # Patrol
    # Civ
    12: 2, # Other merchant
    16: 2, # Barge
    18: 2, # Ferry
    19: 2, # Yacht
    23: 2, # Hovercraft
}

hard2_split = {
    1: 1,  # Other Ship
    # Military
    2: 2,  # Other Military
    10: 2, # Commander
    11: 2, # Aux
    # Civ
    12: 2, # Other merchant
    13: 2, # Container
    14: 2, # RoRo
    19: 2, # Yacht
    21: 2, # Fishing

}

non_other = {
    1: 1,   # Other Ship
    2: 2,   # Other military
    12: 2,  # Other Merchant
}


def create_if_dir_does_not_exist_recursive(dir):

    if os.path.exists(dir):
        print(f"Dir @ {dir} already exists")
        return

    working_path = os.path.sep

    for item in dir.split(os.path.sep):
        working_path = os.path.join(working_path, item)
        if not os.path.exists(working_path):
            os.mkdir(working_path)

    print(f"Created dir @ {working_path}")



def generate_ship_split_files(split_dict, split_name):
    root_dir = '/data/ships/nate/ShipRSImageNet/VOC_Format'
    chips_dir = os.path.join(root_dir, 'chips')
    sets_dir = os.path.join(root_dir, "ImageSets")
    split_dir = os.path.join(root_dir, 'splits', split_name)
    create_if_dir_does_not_exist_recursive(split_dir)

    variants_file = os.path.join(root_dir, 'variants.txt')


    names_to_class = {}
    with open(variants_file, 'r') as variants_file:
        lines = variants_file.readlines()
        for line in lines:
            line = line.rstrip().split(' ')
            names_to_class["".join(line[:-1])] = int(line[-1])


    indicies_to_remove = split_dict.keys()
    id_indicies = set(names_to_class.values()) - set(indicies_to_remove)
    new_indicies_seen = {-1}
    reindexer = {}
    for id_index in sorted(list(id_indicies)):
        next_available_index = max(new_indicies_seen) + 1
        reindexer[id_index] = next_available_index
        new_indicies_seen.add(next_available_index)
        
    print(max(new_indicies_seen))

    sizes = {
        "train": {"ID": 0},
        "test": {"ID": 0, 1: 0, 2: 0},
    }

    id_class_sizes = {}


    chips = set(os.listdir(chips_dir))

    with open(os.path.join(split_dir, "ID_trainset.txt"), "w") as id_train_file, \
	    open(os.path.join(split_dir, "OOD_testset.txt"), "w") as ood_test_file, \
        open(os.path.join(split_dir, "ID_testset.txt"), "w") as id_test_file, \
        open(os.path.join(split_dir, "ID_weights.txt"), 'w') as weights_file:

        with open(os.path.join(sets_dir, "train.txt"), 'r') as trainfile:
            train_file = trainfile.readlines()

        for file in train_file:
            file, _ = os.path.splitext(file.strip())
            matching_chips = {f for f in chips if re.match(fr"{file}_*", f)}
            chips = chips - matching_chips

            for chip in matching_chips:
                name = os.path.splitext(chip)[0].split('_')[-1]
                if name == 'Dock': continue
                heirarchy_class = names_to_class[name]
                level = split_dict.get(heirarchy_class, 0)

                if level == 0:
                    sizes['train']['ID'] += 1
                    new_index = reindexer[heirarchy_class]
                    id_train_file.write(f"{os.path.join(chips_dir, chip)} {new_index}\n")

                    if id_class_sizes.get(new_index, None) is None:
                        id_class_sizes[new_index] = 0 
                    id_class_sizes[new_index] += 1

                else:
                    sizes['test'][level] += 1
                    ood_test_file.write(f"{os.path.join(chips_dir, chip)} {level}\n")


        for key in sorted(list(id_class_sizes.keys())):
            weights_file.write(f"{key} {id_class_sizes[key] / sizes['train']['ID']}\n")

        with open(os.path.join(sets_dir, "val.txt"), 'r') as valfile:
            val_file = valfile.readlines()

            for file in val_file:
                file, _ = os.path.splitext(file.strip())
                matching_chips = {f for f in chips if re.match(fr"{file}_*", f)}
                chips = chips - matching_chips

                for chip in matching_chips:
                    name = os.path.splitext(chip)[0].split('_')[-1]
                    if name == 'Dock': continue
                    heirarchy_class = names_to_class[name]
                    level = split_dict.get(heirarchy_class, 0)

                    if level == 0:
                        sizes['test']['ID'] += 1
                        id_test_file.write(f"{os.path.join(chips_dir, chip)} {reindexer[heirarchy_class]}\n")
                    else:
                        sizes['test'][level] += 1
                        ood_test_file.write(f"{os.path.join(chips_dir, chip)} {level}\n")




    # for file in os.listdir(chips_dir):
    #     tmp = os.path.splitext(file)[0].split('_')
    #     name, file_num = tmp[-1], tmp[-3]

    #     if name == 'Dock': continue

    #     heirarchy_class = names_to_class[name]
    #     level = split_dict.get(heirarchy_class, 0)

    #     if level == 0:
    #         sizes['ID'] += 1
    #     else:
    #         sizes[level] += 1

    print(sizes)
    print(f'Split "{split_name}" has {max(new_indicies_seen)} ID Classes')


generate_ship_split_files(non_other, "non_other")
# generate_ship_split_files(military_split, 'military')
# generate_ship_split_files(civ_split, 'civ')
# generate_ship_split_files(hard1_split, 'hard1')
# generate_ship_split_files(hard2_split, 'hard2')