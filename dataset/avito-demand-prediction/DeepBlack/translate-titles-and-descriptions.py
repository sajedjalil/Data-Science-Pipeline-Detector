import numpy as np
import pandas as pd
import emoji
from googletrans import Translator
import sys
import argparse
import os.path

def parse_input():
    parser = argparse.ArgumentParser()
    # parser.add_argument("-f", help="input file path", required=True) # commenting for kaggle
    parser.add_argument("-f", help="input file path", nargs="?", default="../input/avito-translation-sample/small_sample.csv")
    parser.add_argument("-pre", help="prefix for output files", nargs="?", default = "output")
    parser.add_argument("-title-only", help="pass yes if only title has to be translated", nargs="?", default="no")
    parser.add_argument("-desc-only", help="pass yes if only description has to be translated", nargs="?", default="no")

    args = parser.parse_args()

    if os.path.isfile(args.f) == False:
        sys.exit("Please provide a valid file path")

    return [args.f, args.pre, args.title_only, args.desc_only]


def find_and_replace_emojis(text):
    emoji_counter = 0
    # emoji_unicodes = map( (lambda x: x.encode('utf-8')), emoji.UNICODE_EMOJI.keys())
    emoji_unicodes = emoji.UNICODE_EMOJI.keys()
    for k in emoji_unicodes:
        try:
            if k in text:
                text = text.replace(k, ' ')
                emoji_counter = emoji_counter+1
        except Exception as e:
            print(e)
            print(k)
            print(text)
            raise e
    return [emoji_counter, text]

    
def translate(file_path, prefix, col_name):

    input_csv = pd.read_csv(file_path)
    translator = Translator()
    translations = {}

    num_rows = input_csv.shape[0]

    print("***TRANSLATING "+col_name.upper()+ "***")

    # Translate 20 descriptions at a time
    for i in range(0, num_rows+10, 20):

        print("starting index: " + str(i)+" of "+ str(num_rows))
        orig_text_subset = input_csv[col_name][i:i+20].fillna("None").tolist()
        item_ids = input_csv['item_id'][i:i+20].tolist()

        try:
            translations_subset = translator.translate(orig_text_subset, src='ru', dest='en')
        except Exception as ex:
            translations_subset = []
            for s in orig_text_subset:

                formatted_str = s
                num_emojis = 0
                if type(ex == ValueError):
                    removed_emojis = find_and_replace_emojis(s)
                    num_emojis = removed_emojis[0]
                    formatted_str = removed_emojis[1]

                try:
                    if num_emojis > 0:
                        translations_subset.append(translator.translate(formatted_str, src='ru', dest='en').text + " NUM_EMOJIS: " + str(num_emojis))
                    else:
                        translations_subset.append(translator.translate(formatted_str, src='ru', dest='en').text)
                except:
                    if type(ex == ValueError):
                        translations_subset.append("UNABLE TO TRANSLATE" + " NUM_EMOJIS: " + str(num_emojis))
                    else:
                        translations_subset.append("UNABLE TO TRANSLATE")
            translations.update(dict(zip(item_ids, translations_subset)))
        else:
            translations.update(
                dict(
                    zip(
                        item_ids,
                        list(map( (lambda x: x.text), translations_subset ))
                        )
                    )
            )

    pd.DataFrame(translations.items()).to_csv(prefix +'_' + col_name +'_translation.csv', encoding='utf-8')

def translate_title_and_description(file_path, prefix, title_only, desc_only):
    
    if title_only == 'yes':
        translate(file_path, prefix, "title")
    elif desc_only == 'yes':
        translate(file_path, prefix, "description")
    else:
        translate(file_path, prefix, "title")
        translate(file_path, prefix, "description")



if __name__ == "__main__":
    args = parse_input()
    file_path = args[0]
    prefix = args[1]
    title_only = args[2].lower()
    desc_only = args[3].lower()

    translate_title_and_description(file_path, prefix, title_only, desc_only)
    