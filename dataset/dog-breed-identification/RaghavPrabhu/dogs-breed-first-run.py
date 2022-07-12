#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 07:16:04 2018

@author: raghav
"""
import tensorflow as tf
import sys
import os
import csv

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf


def classify_image(image_path, headers):
    f = open('submit.csv','w')
    writer = csv.DictWriter(f, fieldnames = headers)
    
    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("../input/dogbreedmodel/retrained_labels.txt")]
   
    # Unpersists graph from file
    with tf.gfile.FastGFile("../input/dogbreedmodel/retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    files = os.listdir(image_path)
    with tf.Session() as sess:
         for file in files:
             # Read the image_data
                image_data = tf.gfile.FastGFile(file, 'rb').read()
                # Feed the image_data as input to the graph and get first prediction
                softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

                predictions = sess.run(softmax_tensor, \
                                       {'DecodeJpeg/contents:0': image_data})

                # Sort to show labels of first prediction in order of confidence
                top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
                records = []
                row_dict = {}
                head, tail = os.path.split(image_path)
                row_dict['id'] = tail.split('.')[0]

                for node_id in top_k:
                    human_string = label_lines[node_id]
                    human_string = human_string.replace(" ","_")
                    if(human_string == 'german_short_haired_pointer'):
                        human_string = 'german_short-haired_pointer'
                    if(human_string == 'shih_tzu'):
                        human_string = 'shih-tzu'
                    if(human_string == 'wire_haired_fox_terrier'):
                        human_string = 'wire-haired_fox_terrier'
                    if(human_string == 'curly_coated_retriever'):
                        human_string = 'curly-coated_retriever'
                    if(human_string == 'black_and_tan_coonhound'):
                        human_string = 'black-and-tan_coonhound'
                    if(human_string == 'soft_coated_wheaten_terrier'):
                        human_string = 'soft-coated_wheaten_terrier'  
                    if(human_string == 'flat_coated_retriever'):
                        human_string = 'flat-coated_retriever'    
                    score = predictions[0][node_id]
                    print('%s (score = %.5f)' % (human_string, score))
                    row_dict[human_string] = score
        
                    records.append(row_dict.copy())
                    writer.writerows(records)
                f.close()    

def main():
    test_data = '../input/dog-breed-identification/test/'
    files = os.listdir(test_data)
    
    template_file = open('../input/dogbreedmodel/submit.csv','r')
    d_reader = csv.DictReader(template_file)

    #get fieldnames from DictReader object and store in list
    headers = d_reader.fieldnames
    #print(headers)
    template_file.close()

    #f = open('submit.csv','w')
    #writer = csv.DictWriter(f, fieldnames = headers)
    #writer.writeheader()
    #f.close()
    for file in files:
        classify_image(test_data, headers)
    #f.close()
    

if __name__ == '__main__':
    main()
