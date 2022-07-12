#
#    A very simple way
#    LB ~0.3435 when threshold is 0.2
#

threshold=0.2


#read prior orders
fr = open("../input/order_products__prior.csv", 'r')
fr.readline()# skip header
lines=fr.readlines()
orders={}
orders_reorderer={}
print(len(lines))
for i,line in enumerate(lines):
    datas=line.replace("\n","").split(",")
    order_id=int(datas[0])
    product_id=int(datas[1])
    reorderer=int(datas[3])
    if(order_id not in orders):
        orders[order_id]=[]
        orders_reorderer[order_id]=[]
    orders[order_id].append(product_id)
    orders_reorderer[order_id].append(reorderer)

#read train orders
fr = open("../input/order_products__train.csv", 'r')
fr.readline()# skip header
lines=fr.readlines()
print(len(lines))
for i,line in enumerate(lines):
    datas=line.replace("\n","").split(",")
    order_id=int(datas[0])
    product_id=int(datas[1])
    reorderer=int(datas[3])
    if(order_id not in orders):
        orders[order_id]=[]
        orders_reorderer[order_id]=[]
    orders[order_id].append(product_id)
    orders_reorderer[order_id].append(reorderer)
  
#read orders
fr = open("../input/orders.csv", 'r')

#output submission
outcsv = open("submission.csv", 'w')
outcsv.writelines("order_id,products"+"\n")
fr.readline()# skip header
lines=fr.readlines()
users={}
print(len(lines))
for i,line in enumerate(lines):
    datas=line.replace("\n","").split(",")
    order_id=int(datas[0])
    user_id=int(datas[1])
    eval_set=(datas[2])
    order_number=int(datas[3])
    if(user_id not in users):
        users[user_id]={}
    if(eval_set=="prior"):
        users[user_id][order_number]=order_id
    elif(eval_set=="train"):
        users[user_id][order_number]=order_id
    if(eval_set=="test"):
        users[user_id]["test"]=order_id

print("Start predicting...")
for user_id in users:
    #skip this user if he/she doesn't need to be predict
    if("test" not in users[user_id]):
        continue
    products_reorderer={}
    products_firstbuy_index={}
    products_reorderer_rate={}
    test_order_number=users[user_id]["test"]
    i=0
    for order_number in users[user_id]:
        order_id=users[user_id][order_number]
        if(order_number=="test"):
            continue
        for index,product_id in enumerate(orders[order_id]):
            if(product_id not in products_firstbuy_index):
                products_firstbuy_index[product_id]=i
            if(product_id not in products_reorderer):
                products_reorderer[product_id]=0
            if(orders_reorderer[order_id][index]==1):
                products_reorderer[product_id]+=1
        i+=1

    for product_id in products_reorderer:
        reorder_len=len(users[user_id])-2
        # minus 2 : 1 for the first time buy, 1 for the test order
        if(reorder_len>0):
            products_reorderer_rate[product_id]=products_reorderer[product_id]/float(reorder_len)
        else:
            products_reorderer_rate[product_id]=0
    predict_reorderer_products=[]
    # assume that if one item's reorder rate > 50%
    # then it will be in his/her next order
    for product_id in products_reorderer:
        p_threshold=products_reorderer_rate[product_id]
        if(p_threshold>1):
            print(p_threshold)
        if(p_threshold>=threshold):
            predict_reorderer_products.append(str(product_id))
    # if no product will be reorderer
    if(len(predict_reorderer_products)==0):
        predict_reorderer_products.append("None")
        
    reordered_str=' '.join(predict_reorderer_products)
    outcsv.writelines(str(test_order_number)+","+reordered_str+"\n")
