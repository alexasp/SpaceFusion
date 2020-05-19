from data_prepare import build_vocab, text2num
from shutil import copyfile
from random import shuffle
import numpy as np 

gen_data_path = 'data/generated/ref_10'
output_path = 'data/reddit'

# build_vocab(gen_data_path, fname='2011-01.tsv')
# copyfile(gen_data_path + '/vocab.txt', output_path + '/vocab.txt')

# text2num(gen_data_path, '2011-01', path_vocab=output_path + '/vocab.txt')
# copyfile(gen_data_path + '/2011-01.num', output_path + '/2011-01.num')

lines = []
with open(output_path + '/2011-01.num') as full_file:
    for line in full_file:
        if line.strip():
            lines.append(line)

shuffle(lines)

train, validate, test = np.split(lines, [int(.90*len(lines)), int(.95*len(lines))])

with open(output_path+'/train.num', 'w') as outfile:
    for line in train:
        outfile.write(line)

with open(output_path+'/vali.num', 'w') as outfile:
    for line in validate:
        outfile.write(line)

with open(output_path+'/test.num', 'w') as outfile:
    for line in test:
        outfile.write(line)
