import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

train_df = pd.read_csv('../input/train.csv')
submission_df = pd.read_csv('../input/test.csv')

train_comments_orig = train_df['comment_text']
submission_comments = submission_df['comment_text']

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train_comments, test_comments, train_true, test_true = train_test_split(train_comments_orig, train_df[labels])
test_true_matrix = test_true.as_matrix()

print(len(train_comments))
print(len(train_true))
print(len(test_comments))
print(len(test_true))

# HELPER CLASSES

class CommentsEmbedder():
    
    def __init__(self, fit_comments):
        self.fit_comments = fit_comments
        self.num_words = 10000

        self.vectorizer = vectorizer = TfidfVectorizer(
                                        analyzer='word', 
                                        sublinear_tf=True,
                                        strip_accents='unicode',
                                        token_pattern=r'\w{1,}',
                                        stop_words='english',
                                        ngram_range=(1, 3),
                                        max_features=self.num_words)
        self.tfidf = self.vectorizer.fit(self.fit_comments)
        
    '''transform array of comments to tfidf matrix'''
    def transform(self, comments):
        sparse = self.tfidf.transform(comments)
        return sparse.todense()
        
class CommentData():
    
    def __init__(self, comments, y_true=None):
        
        self.comments = comments
        self.y_true = y_true
        self.i = 0
        self.do_next_batch = True
        
        
    
    def next_batch(self,batch_size):
        if self.i + batch_size >= len(self.comments):
            new_i = len(self.comments) + 1
            self.do_next_batch = False
        else:
            new_i = self.i + batch_size
        
        batch_x = self.comments[self.i:new_i]
        
        if self.y_true is not None:
            batch_y = self.y_true[self.i:new_i].as_matrix()
            self.i = new_i
            return batch_x, batch_y    
        else:
            self.i = new_i
            return batch_x
        
# HELPER FUNCTIONS

def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

def conv1d(x, W):
    # x is input tensor --> [batch, num_words, in_channels]
    # W is the kernel --> [filter width, in_channels, out_channels]
    return tf.nn.conv1d(x,W, stride=1, padding='SAME')

def max_pool(x):
    # x is input tensor --> [batch, num_words, in_channels]
    return tf.nn.pool(x, window_shape=[1], pooling_type='MAX', padding='SAME')

def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[2]])
    
    return tf.nn.relu(conv1d(input_x, W) + b)

def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    
    return tf.matmul(input_layer,W) + b
    
#### EMBED

print('Fitting vectorizer...')
        
comment_embedder = CommentsEmbedder(train_comments_orig)

print('Transforming test comments...')

test_comment_matrix = comment_embedder.transform(test_comments)

# GRAPH PLACEHOLDERS AND SOME HYPERPARAMETERS

batchSize = 15
numClasses = len(labels)
iterations = 10635

tf.reset_default_graph()

print('Max possible number of iterations: {}'.format(int(len(train_comments)/batchSize)))

input_data = tf.placeholder(tf.float32, shape=[None, comment_embedder.num_words])
y_true = tf.placeholder(tf.float32, shape=[None, numClasses])
hold_prob = tf.placeholder(tf.float32)


# LAYERS

convo_1_num_features = 10
convo_1_filter_width = 2

convo_2_num_features = 20
convo_2_filter_width = 2

convo_input = tf.reshape(input_data, [-1, comment_embedder.num_words ,1]) 

convo_1 = convolutional_layer(convo_input, shape=[convo_1_filter_width, 1, convo_1_num_features])

convo_1_pooling = max_pool(convo_1)

convo_2 = convolutional_layer(convo_1_pooling, shape=[convo_2_filter_width, convo_1_num_features, convo_2_num_features])

#convo_2 = convolutional_layer(convo_input, shape=[convo_2_filter_width, 1, convo_2_num_features])

convo_2_pooling = max_pool(convo_2)

convo_2_flat = tf.reshape(convo_2_pooling, shape=[-1, comment_embedder.num_words*convo_2_num_features])


# DROPOUT

dropout = tf.nn.dropout(convo_2_flat, keep_prob=hold_prob)


normal_full = normal_full_layer(dropout, numClasses)
y_pred = tf.sigmoid(normal_full)


# LOSS FUNCTION

loss = tf.reduce_mean(tf.losses.log_loss(y_true, y_pred))
train = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)


# SUBMISSION PREDICTER

predict = y_pred


# TRAIN

init = tf.global_variables_initializer()

with tf.Session() as sess:
    
    data = CommentData(train_comments, train_true)
    
    sess.run(init)
    
    for i in range(iterations):
        
        if data.do_next_batch == False:
            break
        
        batch_x , batch_y = data.next_batch(batchSize)
        
        batch_x = comment_embedder.transform(batch_x)
        
        sess.run(train,feed_dict={input_data:batch_x, y_true:batch_y, hold_prob:0.8})
        
        # PRINT OUT A MESSAGE EVERY 100 STEPS
        if i%100 == 0:
            
            print('Currently on step {}'.format(i))
            print('Test Set Loss:')
            
            # get random sample of 5000 from test set
            test_indexes = np.random.randint(0, len(test_comments)-1, size=5000)
            
            test_comment_matrix_5000 = [test_comment_matrix[i] for i in test_indexes]
            test_comment_matrix_5000 = np.reshape(test_comment_matrix_5000, [5000, -1])
            
            test_true_5000 = [test_true_matrix[i] for i in test_indexes]
            test_true_5000 = np.reshape(test_true_5000, [5000, -1])
            
            print(sess.run(loss,feed_dict={input_data:test_comment_matrix_5000, y_true:test_true_5000, hold_prob:1.0}))
            print('\n')
     
    print('Finished training, making predictions...')   
    
    # overwrite train data to save memory
    data = None
    submission_data = CommentData(submission_comments, None)
    
    # clear test matrix
    test_comment_matrix = None
    
    last_submission_ix = 0
    
    while submission_data.do_next_batch == True:
        
        batch_x = submission_data.next_batch(batchSize)
        
        batch_x = comment_embedder.transform(batch_x)
        
        if submission_data.i%10000 == 0:
            print('On prediction {}'.format(submission_data.i))

        submission_pred = sess.run(predict,feed_dict={input_data:batch_x, hold_prob:1.0})
        
        try:
          pred_df
        except NameError:
          pred_df = pd.DataFrame(data=submission_pred, index=submission_df['id'][:submission_data.i], columns=labels)
          last_submission_ix = submission_data.i
        else:
          pred_df = pred_df.append(pd.DataFrame(data=submission_pred, index=submission_df['id'][last_submission_ix:submission_data.i], columns=labels))
          last_submission_ix = submission_data.i

    pred_df.to_csv('submission.csv')