# NAI

import os

# This is where we are actually defining the ood data which will be split out
# dict: "Variant-Name":OOD-Group
split1_dict = {
	"DH-82":1,
	"DHC-1":1,
	"DHC-6":1,
	"DHC-8-100":1,
	"DHC-8-300":1,
	"Fokker 100":1,
	"Fokker 50":1,
	"Fokker 70":1,
	"747-100":2,
	"747-200":2,
	"747-300":2,
	"747-400":2,
	"A340-200":2,
	"A340-300":2,
	"A340-500":2,
	"A340-600":2,
	"737-400":3,
	"737-900":3,
	"767-300":3,
	"A320":3,
	"E-170":3,
}

split2_dict = {
	"ATR-42":1,
	"ATR-72":1,
	"E-170":1,
	"E-190":1,
	"E-195":1,
	"EMB-120":1,
	"ERJ 135":1,
	"ERJ 145":1,
	"Embraer Legacy 600":1,
	"767-200":2,
	"767-300":2,
	"767-400":2,
	"A330-200":2,
	"A330-300":2,
	"MD-80":2,
	"MD-87":2,
	"737-300":3,
	"737-700":3,
	"A318":3,
	"A340-600":3,
}

split3_dict = {
	"CRJ-200":1,
	"CRJ-700":1,
	"CRJ-900":1,
	"Cessna 172":1,
	"Cessna 208":1,
	"Cessna 525":1,
	"Cessna 560":1,
	"Challenger 600":1,
	"757-200":2,
	"757-300":2,
	"777-200":2,
	"777-300":2,
	"DHC-8-100":2,
	"DHC-8-300":2,
	"ERJ 135":2,
	"ERJ 145":2,
	"737-200":3,
	"747-400":3,
	"A319":3,
	"A340-600":3,
	"E-195":3,
	"MD-87":3,
}

split4_dict = {
	"An-12":1,
	"Beechcraft 1900":1,
	"DR-400":1,
	"Il-76":1,
	"Model B200":1,
	"SR-20":1,
	"Yak-42":1,
	"747-100":2,
	"747-200":2,
	"747-300":2,
	"747-400":2,
	"A318":2,
	"A319":2,
	"A320":2,
	"A321":2,
	"737-500":3,
	"737-600":3,
	"767-200":3,
	"A340-500":3,
	"E-170":3,
}
   
def generate_fgcv_aircraft_split_files(split_dict, split_name):

	root_dir = "/raid/inkawhna/WORK/fgvc-aircraft-2013b/data" # absolute path
	orig_train_file = root_dir+"/images_variant_trainval.txt"
	orig_test_file = root_dir+"/images_variant_test.txt"
	split_dict_keys = list(split_dict.keys())
	
	# Create class names LUT
	variants = []
	for l in open(root_dir+"/variants.txt","r"):
		variants.append(l.rstrip())
	for k in split_dict_keys:
		assert(k in variants)
	id_variants = [v for v in variants if v not in split_dict_keys]
	print("# Orig Variants: ",len(variants))
	print("# ID Variants: ",len(id_variants))
	print("# OOD Variants: ",len(split_dict_keys))

	# Parse original train file and write to id_train and ood_test
	id_train_file = open("{}/splits/{}/ID_trainset.txt".format(root_dir, split_name),"w")
	ood_test_file = open("{}/splits/{}/OOD_testset.txt".format(root_dir, split_name),"w")
	for l in open(orig_train_file,"r"):
		l = l.rstrip()
		print(l)
		img_id = l[:7]
		variant = l[8:]
		img_name = "{}/images/{}.jpg".format(root_dir, img_id)
		assert(os.path.exists(img_name))
		if variant in split_dict_keys:
			print("OOD test-> {} {}".format(img_name, split_dict[variant]))
			ood_test_file.write("{} {}\n".format(img_name, split_dict[variant])) # write: "/pth/to/img.jpg ood_group_num"
		elif variant in id_variants:
			print("ID train-> {} {}".format(img_name, id_variants.index(variant)))
			id_train_file.write("{} {}\n".format(img_name, id_variants.index(variant))) # write: "/pth/to/img.jpg id_cls"
		else:
			exit("Bad variant name!")
	
	id_train_file.close()

	# Same loop over orig_test_file
	id_test_file = open("{}/splits/{}/ID_testset.txt".format(root_dir, split_name),"w")
	for l in open(orig_test_file,"r"):
		l = l.rstrip()
		print(l)
		img_id = l[:7]
		variant = l[8:]
		img_name = "{}/images/{}.jpg".format(root_dir, img_id)
		assert(os.path.exists(img_name))
		if variant in split_dict_keys:
			print("OOD test-> {} {}".format(img_name, split_dict[variant]))
			ood_test_file.write("{} {}\n".format(img_name, split_dict[variant])) # write: "/pth/to/img.jpg ood_group_num"
		elif variant in id_variants:
			print("ID test-> {} {}".format(img_name, id_variants.index(variant)))
			id_test_file.write("{} {}\n".format(img_name, id_variants.index(variant))) # write: "/pth/to/img.jpg id_cls"
		else:
			exit("Bad variant name!")

	id_test_file.close()
	ood_test_file.close()
	

	# Write class names file
	class_names_file = open("{}/splits/{}/class_names.txt".format(root_dir, split_name),"w")
	print("Writing ID class names file...")
	for i,c in enumerate(id_variants):
		print("{} {}".format(str(i).zfill(3), c))
		class_names_file.write("{} {}\n".format(str(i).zfill(3), c))
	class_names_file.close()

	print("Done writing split files!")
	return
		
		
generate_fgcv_aircraft_split_files(split1_dict, "split1")
#generate_fgcv_aircraft_split_files(split2_dict, "split2")
#generate_fgcv_aircraft_split_files(split3_dict, "split3")
#generate_fgcv_aircraft_split_files(split4_dict, "split4")


 
print("Done!")
