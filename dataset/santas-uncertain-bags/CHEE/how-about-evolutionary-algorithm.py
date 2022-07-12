# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import numpy as np
import math
import random

print("Start")

class Box:
    def __init__(self):
        self.weight = 0
        self.score = 0
        self.item = []
    def add_item(self, gift_type):
        self.item.append(gift_type)
    def remove_random_item(self):
        if len(self.item) == 0: return 9999
        index = math.floor(random.random() * len(self.item))
        return self.item.pop(index)
    def calculate_total_weight(self):
        iteration = 100
        w = 0
        for i in self.item:
            for j in range(iteration):
                w += self.check_weight(i)
        self.weight = w / iteration
        if (self.weight <= 50) :  self.score = self.weight
        else :                    self.score = 0
    def check_weight(self, item_number): # TODO: Need to fix these value for each generation
        if 2 <= item_number <= 1001 :  # house 1000
            return max(0, np.random.normal(5, 2, 10)[0])
        elif 1002 <= item_number <= 2101:  # ball 1100
            return max(0, 1 + np.random.normal(1, 0.3, 1)[0])
        elif 2012 <= item_number <= 2601:  # bike 500
            return max(0, np.random.normal(20, 10, 1)[0])
        elif 2602 <= item_number <= 3601:  # train 1000
            return max(0, np.random.normal(10, 5, 1)[0])
        elif 3602 <= item_number <= 3767:  # coal 163
            return 47 * np.random.beta(0.5, 0.5, 1)[0]
        elif 3768<= item_number <= 4967:  # book 1200
            return np.random.chisquare(2, 1)[0]
        elif 4968 <= item_number <= 5967:  # doll 1000
            return np.random.gamma(5, 1, 1)[0]
        elif 5968 <= item_number <= 6967:  # block 1000
            return np.random.triangular(5, 10, 20, 1)[0]
        elif 6968 <= item_number <= 7167:  # gloves 200
            return 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0]
        else:
            print("ERROR !!")
            return 0
    def print(self):
        output = ""
        for i in self.item:
            output += self.get_name(i)
        output += "\n"
        return output
    def get_name(self, item_number):
        if 2 <= item_number <= 1001 :  # house 1000
            return "horse_%d " % (item_number - 2)
        elif 1002 <= item_number <= 2101:  # ball 1100
            return "ball_%d " % (item_number - 1002)
        elif 2102 <= item_number <= 2601:  # bike 500
            return "bike_%d " % (item_number - 2102)
        elif 2602 <= item_number <= 3601:  # train 1000
            return "train_%d " % (item_number - 2602)
        elif 3602 <= item_number <= 3767:  # coal 163
            return "coal_%d " % (item_number - 3602)
        elif 3768<= item_number <= 4967:  # book 1200
            return "book_%d " % (item_number - 3768)
        elif 4968 <= item_number <= 5967:  # doll 1000
            return "doll_%d " % (item_number - 4968)
        elif 5968 <= item_number <= 6967:  # block 1000
            return "blocks_%d " % (item_number - 5968)
        elif 6968 <= item_number <= 7167:  # gloves 200
            return "gloves_%d " % (item_number - 6968)
        else:
            print("ERROR !!")
            return "ERROR !!"
class Individual():
    def __init__(self):
        self.generation = 0
        self.fitness = 0
        self.number_of_overweight_before = 0
        self.number_of_overweight_after = 0
        self.box = []
        self.available_present_number = list(range(2, 7168))
        for i in range(1000):
            self.box.append(Box())
    def calculate_fitness(self):
        last_fitness = self.fitness
        self.fitness = 0
        self.number_of_overweight_after = 0
        for i in range(1000):
            self.box[i].calculate_total_weight()
            if len(self.box[i].item) > 0 and self.box[i].score == 0:  # OVERWEIGHT !
                    self.number_of_overweight_after += 1
            self.fitness += self.box[i].score
        improvement = self.fitness - last_fitness
        if self.generation % 1 == 0:
            print("gen: %5d  fit: %5d  imprvd: %4d  OW_b4: %2d  OW_aftr: %2d  gift_left: %4d" % (self.generation, self.fitness, improvement, self.number_of_overweight_before, self.number_of_overweight_after, len(self.available_present_number)))
        return improvement
    def mutate(self):
        mutate_rate = 10
        algorithm = 1
        self.generation += 1
    #if algorithm == 1:
        self.number_of_overweight_before = 0

        for i in range(mutate_rate):
            add_or_remove = round(random.random())
            which_box = math.floor(random.random()*1000)
            if add_or_remove == 0 and len(self.available_present_number) > 0:
                self.box[which_box].add_item(self.get_gift())
            else:
                removed_item_number = self.box[which_box].remove_random_item()
                if removed_item_number != 9999:
                    self.return_gift(removed_item_number)

        for i in range(1000):
            if len(self.box[i].item) > 0 and self.box[i].score == 0:  # OVERWEIGHT !
                self.number_of_overweight_before += 1
                self.handle_overweight(i)
            elif self.box[i].score < 25 and len(self.available_present_number) > 0: # UNDERWEIGHT #(2000-self.generation)/50
                self.box[i].add_item(self.get_gift())

            if len(self.box[i].item) < 3: # UNDERCOUNT < 3
                for j in range(3 - len(self.box[i].item)):
                    if len(self.available_present_number) > 0:
                        self.box[i].add_item(self.get_gift())
    #else:
        #for i in range(1000):
            #if len(self.box[i].item) != 0:
                #print(i, "-", self.box[i].item)
    def get_gift(self):
        gift_index = math.floor(random.random()*len(self.available_present_number))
        # Swap with last one
        choosen_number = self.available_present_number[gift_index]
        self.available_present_number.pop(gift_index)
        return choosen_number
    def return_gift(self,removed_item_number):
        self.available_present_number.append(removed_item_number)
    def handle_overweight(self,which_box):
        # remove 1 item
    #while (len(self.box[which_box].item) > 0 and self.box[which_box].score == 0):
        removed_item_number = self.box[which_box].remove_random_item()
        if removed_item_number != 9999:
            self.return_gift(removed_item_number)
            #print(removed_item_number)
        else: print("ERROR #1 !")

        #self.box[which_box].calculate_total_weight()

        #if len(self.box[which_box].item) > 0 and self.box[which_box].score == 0:
            #self.number_of_overweight_after += 1

    def next_generation(self):
        old_box = self.box
        self.mutate()
        if self.calculate_fitness() < 0:
            self.box = old_box

i1 = Individual()

for i in range(600):
    i1.next_generation()


n = "5"
with open("submission_%s.csv" % n, 'w+') as submission_file:
    submission_file.write('Gifts\n')
    for i in range(1000):
        submission_file.write(i1.box[i].print())


print("END")





print ("to be continue, good night")