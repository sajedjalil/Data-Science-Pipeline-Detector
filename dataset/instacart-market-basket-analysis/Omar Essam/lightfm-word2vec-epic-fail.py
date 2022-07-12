"""
Disclaimer: This kernel is an absolute fail.

I'm only sharing this because it can be a starting point for using LightFM (A Recommendation algorithm) for this competition.

The output of this script is the top 5 recommendations predicted by LightFM, however LightFM does NOT predict items a user
has already 'rated', so in that sense it fails in this competition that is about which products will be reordered.

However if there's a way this behavior can be altered then we can use it as a viable solution.

Another point is that this is user based collaborative filtering, IMHO we should use item based solutions.

An important point is that this script will use the output from the Word2Vec model that we train on
products, which includes information already about products' recommendations.
You can find my kernel here explaining it and producing the model:
https://www.kaggle.com/omarito/word2vec-for-products-analysis-product2vec/notebook

If you have any ideas to improve this please share them, we're all learning after all.
For reference, LightFM:
https://github.com/lyst/lightfm/
"""
import pandas as pd
import numpy as np
import tqdm

from lightfm import LightFM

from sklearn.feature_extraction.text import CountVectorizer

from gensim.models import Word2Vec
from scipy import sparse

def flatten(products):
	output = []
	for product in products:
		if(type(product) is str):
			output.extend(product.split(" "))
	return " ".join(output)

def prod2vec(prod):
    if(prod in list(Product2Vec.wv.vocab.keys())):
        return Product2Vec[prod]
    else:
        return np.zeros((100,))

train_orders = pd.read_csv("../input/order_products__train.csv")
prior_orders = pd.read_csv("../input/order_products__prior.csv")
orders = pd.read_csv("../input/orders.csv")
products = pd.read_csv("../input/products.csv")
aisles = pd.read_csv("../input/aisles.csv")
departments = pd.read_csv("../input/departments.csv")


train = orders[orders['eval_set'] == 'train']
test = orders[orders['eval_set'] == 'test']
prior = orders[orders['eval_set'] == 'prior']

train_orders['product_id'] = train_orders['product_id'].apply(lambda x: str(x))
prior_orders['product_id'] = prior_orders['product_id'].apply(lambda x: str(x))


# Our goal here is to construct a huge (sparse) matrix that is of cardinality User * Product.
# This matrix specifies for each User which products did they ever order.

all_orders = train_orders.append(prior_orders)
all_orders = all_orders.groupby("order_id").apply(lambda x: " ".join(x['product_id']))
all_orders.name = "products"
orders = orders.join(all_orders, on='order_id')
user_products = orders.groupby('user_id')['products'].apply(lambda x: list(x))
user_products = user_products.apply(flatten)

product_ids = products['product_id'].apply(lambda x: str(x)).unique()


# Count vectorizer will build a Bag of Words model that will label each product the user has ordered
cv = CountVectorizer(vocabulary=product_ids)

user_products_matrix = cv.fit_transform(user_products)

# This is the model from the Word2vec kernel
Product2Vec = Word2Vec.load("instacart_products_vectors.model")

product_vectors = pd.DataFrame(data={"Vectors": product_ids})
product_vectors["Vectors"] = product_vectors["Vectors"].apply(prod2vec)

product_vectors.append(pd.DataFrame(data={"Vectors": np.zeros(1)}))
product_vectors = product_vectors.shift(1)
product_vectors.iloc[0]["Vectors"] = np.zeros((100,))
product_vectors["Vectors"] = product_vectors["Vectors"].apply(lambda x: x.astype(np.float32))

product_vectors = np.array(product_vectors["Vectors"].tolist())
product_vectors = sparse.csr_matrix(product_vectors)

# Training LightFM on the User Products matrix
NUM_THREADS = 4
NUM_COMPONENTS = 30
NUM_EPOCHS = 5
ITEM_ALPHA = 1e-6

model = LightFM(loss='warp',
                item_alpha=ITEM_ALPHA,
               no_components=NUM_COMPONENTS)

model = model.fit(user_products_matrix, epochs=NUM_EPOCHS, num_threads=NUM_THREADS, item_features=product_vectors)

results = []

# Predict for each user (If you know how to vectorize it would be more efficient)
for user_id in tqdm.tqdm(test['user_id'].unique()):
	recom = model.predict(user_id, np.arange(len(product_ids)), item_features=product_vectors, num_threads=NUM_THREADS)
	recom = pd.Series(recom)
	recom.sort_values(ascending=False, inplace=True)
	if(len(results) == 0):
		results = np.array(recom.iloc[0:5].index.values)
	else:
		results = np.vstack((results, recom.iloc[0:5].index.values))

# A Work around, we can only submit predictions for orders not users
# So since we have a 1 to 1 mapping in the test set, we can use that
# to replace user ids with order ids.

results_df = pd.DataFrame(data=results)
results_df = results_df.apply(lambda x: " ".join(x.astype(str)), raw=True, axis=1).to_frame()
ids = test['user_id'].unique()
results_df['order_id'] = ids

user2order = test[['user_id', 'order_id']]
user2order.set_index('user_id', inplace=True)
user2order = user2order.to_dict()['order_id']

results_df['order_id'] = results_df['order_id'].map(user2order)
results_df.set_index('order_id', inplace=True)
results_df.columns = ["products"]
results_df.to_csv("instacart.csv")
