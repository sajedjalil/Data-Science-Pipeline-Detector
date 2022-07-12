# for each gift, quantify happiness for each good kid

root_dir = '../input/'

def create_gift2child_happiness_map():
    gift_file = open(root_dir + 'gift_goodkids_v2.csv', 'r')
    gift_happiness_file = open('gift2child_happiness.csv', 'w')
    line_count = 0
    for line in gift_file:
        tokens = line.split(',')
        gift_id = int(tokens[0])
        good_kids_count = len(tokens)
        line_count += 1
        for t in range(1, good_kids_count):
            good_kid_id = int(tokens[t])
            happiness = 2*(good_kids_count - (t-1)) / good_kids_count # normalized
            gift_happiness_file.write(str(gift_id) +',' + str(good_kid_id) + ',' + str(happiness) + '\n')
    print('total lines processed in gift file:' + str(line_count))
    gift_happiness_file.close()

# for each child, quantify happiness for each gift on wishlist
def create_child2gift_happiness_map():
    child_file = open(root_dir + 'child_wishlist_v2.csv', 'r')
    child_happiness_file = open('child2gift_happiness.csv', 'w')
    line_count = 0
    for line in child_file:
        tokens = line.split(',')
        child_id = int(tokens[0])
        wishlist_count = len(tokens)
        line_count +=1
        for t in range(1, wishlist_count):
            gift_id = int(tokens[t])
            happiness = 2*(wishlist_count - (t-1)) / wishlist_count # normalized
            child_happiness_file.write(str(child_id) +',' + str(gift_id) + ',' + str(happiness) + '\n')
    print('total lines processed in child file:' + str(line_count))
    child_happiness_file.close()

if __name__ == '__main__':
    
    print('download and uncomment the lines two functions insidie __main__ before executing')
    print('note that this will produce two huge (2g) files') 
    
    # create_child2gift_happiness_map()
    # create_gift2child_happiness_map()