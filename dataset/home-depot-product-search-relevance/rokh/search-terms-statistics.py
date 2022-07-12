import csv
from collections import Counter

def read_file(filename):
    with open(filename, errors='ignore') as f:
        reader = csv.DictReader(f)
        for r in reader:
            yield r


def print_stat(filename):
    search_terms = Counter()
    for line in read_file('../input/' + filename):
        search_terms[line['search_term']] += 1
    
    print('Search term stat for file {}:'.format(filename))
    print('\tTotal search terms:', len(search_terms))
    print('\tMost common search terms:')
    for search_term, count in search_terms.most_common(10):
        print('\t\t{} {}'.format(str(count).rjust(4), search_term))
    print('\tLeast common search terms:')
    for search_term, count in search_terms.most_common()[-10:]:
        print('\t\t{} {}'.format(str(count).rjust(4), search_term))
    
    return search_terms

tr = set(print_stat('train.csv'))
te = set(print_stat('test.csv'))

print('TR-TE:', len(tr-te))
print('TE-TR:', len(te-tr))
print('TE|TR:', len(te|tr))
print('TE&TR:', len(te&tr))