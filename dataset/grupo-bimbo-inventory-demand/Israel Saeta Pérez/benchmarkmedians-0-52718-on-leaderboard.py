# Uses the entire dataset to run and compute the required means.

from collections import defaultdict

prod_demanda_sum = defaultdict(int)
prod_count = defaultdict(int)

prod_client_demanda_sum = defaultdict(int)
prod_client_count = defaultdict(int)

global_sum = 0
global_count = 0

# We don't use Python CSV module because it is significatively slower
# than plain read and split

# create rules according to training
with open('../input/train.csv') as train_fh:
    _ = next(train_fh)  # skip header
    for i, tx in enumerate(train_fh):
        if i % 1000000 == 0:
            print('Read {} lines...'.format(i))

        tx_parts = tx.split(',')
        demanda = int(tx_parts[-1])
        client = int(tx_parts[4])
        prod = int(tx_parts[5])
        prod_demanda_sum[prod] += demanda
        prod_count[prod] += 1

        prod_client = (prod, client)
        prod_client_demanda_sum[prod_client] += demanda
        prod_client_count[prod_client] += 1

        global_sum += demanda
        global_count += 1

    global_mean = round(global_sum / float(global_count))

# counters to know how many predictions were performed with each type
# of rule
prod_client_applied = 0
prod_applied = 0
global_applied = 0

# apply those rules to test
with open('../input/test.csv') as test_fh, \
        open('means_submission.csv', 'w') as submission_fh:
    submission_fh.write('id,Demanda_uni_equil\n')  # write submission header
    _ = next(test_fh)  # skip header in test file
    for i, tx in enumerate(test_fh):
        if i % 1000000 == 0:
            print('Wrote {} lines...'.format(i))

        tx_parts = tx.split(',')
        tx_id = int(tx_parts[0])
        client = int(tx_parts[5])
        prod = int(tx_parts[6])
        # Assign computed means in this order of preference:
        #  - (product, client)
        #  - product
        #  - global
        prod_client = (prod, client)
        if prod_client in prod_client_demanda_sum:
            mean_demanda = round(prod_client_demanda_sum[prod_client] /
                                 float(prod_client_count[prod_client]))
            prod_client_applied += 1
        elif prod in prod_demanda_sum:
            mean_demanda = round(prod_demanda_sum[prod] /
                                 float(prod_count[prod]))
            prod_applied += 1
        else:
            mean_demanda = global_mean
            global_applied += 1

        submission_fh.write('{},{:.0f}\n'.format(tx_id, mean_demanda))

print('Product client: {}'.format(prod_client_applied))
print('Product: {}'.format(prod_applied))
print('Global: {}'.format(global_applied))
