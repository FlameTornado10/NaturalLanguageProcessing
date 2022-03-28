from tabulate import tabulate
from scipy.stats import entropy
import pandas as pd
import numpy as np
import json_lines as jls
import csv


# generate space and letters
letters = ['space']
for index in range(26):
    letters.append(chr(ord('A') + index))


def write_times_to_csv():
    times = []
    ALL = []
    for i in range(27):
        times.append(0)
    count = 0
    ALL.append(letters)
    with open('contents.jl', 'rb') as f:
        for item in jls.reader(f):
            for char in item['p']:
                if count % 500000 == 0 and count > 0:
                    temp = times[:]
                    ALL.append(temp)
                if char == ' ':
                    count = count + 1
                    times[0] = times[0] + 1
                elif char.isalpha():
                    count = count + 1
                    char = char.upper()
                    if len(char) > 1:
                        continue
                    index = ord(char) - ord('A') + 1
                    if 26 >= index > 0:
                        times[index] = times[index] + 1
    temp = times[:]
    ALL.append(temp)
    with open('times.csv', 'w') as f:
        write = csv.writer(f)
        write.writerows(ALL)
    print(np.shape(ALL))


prob = []


# calculate probability and entropy
def cal_prob_and_entropy():
    csvfile = pd.read_csv('times.csv')
    alltimes = np.array(csvfile)
    for i, times in enumerate(alltimes):
        times = times.tolist()
        sum_of_times = sum(times)
        print(f"{(i + 1)*500000}letters\t","entropy:", entropy(times, base=2).round(4))



# write_times_to_csv()
cal_prob_and_entropy()
