# python check_zip_md5.py <directory containing zips>
import os,sys, hashlib

# WARNING: Produced on windows system - please scream at me if I've inadvertently done something platform specific here.

# (probably) correct md5 hashes grabbed from here:
# https://www.kaggle.com/c/avito-duplicate-ads-detection/forums/t/20877/file-checksums
# acknowledgements to:
# FernandoProcy
# David Tran
hash_dict = {"489dab78a6afada8654ac0947eeffc64": "Images_0.zip",
    "3b874896ed26021a8585170b1f1dd0d4": "Images_1.zip",
    "5677fdd58dd0c75bec1a2618f9ddd16f": "Images_2.zip",
    "66213fe3b8e4923410309daf7214044e": "Images_3.zip",
    "d7cdac13e99cc93a4210a9f1799b1139": "Images_4.zip",
    "15357acfe956b6105405329fb1e7c9de": "Images_5.zip",
    "d2d87c47aef9924e8be5304432e42af4": "Images_6.zip",
    "541e9dd21bc5a42af5880f2e3e39eec8": "Images_7.zip",
    "74fd600b56909c0a88059f338c4e9040": "Images_8.zip",
    "dc65300a9b969204cd7ead5b274c1d82": "Images_9.zip",
    "ea2e8b85ea83c2ec019e1ffe42817110": "ItemInfo_test.csv.zip",
    "b743272b4742be9689c4da57ba1f60ae": "ItemInfo_train.csv.zip",
    "5856a92e56bd994f819e80667019ab76": "ItemPairs_test.csv.zip",
    "0fbbe128e2899d5b8d7d24d0da3995b3": "ItemPairs_train.csv.zip"}

if len(sys.argv) < 2:
    raise ValueError("Missing argument: expected call format - python check_zip_md5.py <directory containing zips>")

# Assuming here that all zips you want checked are in one directory, supplied via the command-line
zip_directory = sys.argv[1]
   
# Unnecessary codewise - but I didn't want to fiddle too much with the text I copy-pasted from the forum
hash_dict_inv = {v: k for k, v in hash_dict.items()}

for item in os.listdir(zip_directory):
    thing = os.path.join(zip_directory, item)
    if os.path.isfile(thing):
        filename = thing.split(os.path.sep)[-1]
        extension = filename.split(".")[-1]
        if ( extension == 'zip' ) and ( filename in hash_dict_inv.keys() ): # only for zip files
 
            md5 = hashlib.md5()
            with open(thing,'rb') as f: 
                for chunk in iter(lambda: f.read(128*md5.block_size), b''): 
                    md5.update(chunk)

            print(filename,hash_dict_inv[filename])
            if md5.hexdigest() == hash_dict_inv[filename]:
                print("checks out!")            
