import numpy as np
import pandas as pd
import tensorflow as tf
import kagglegym
import tqdm

def preprocess(data):
    features = data
    data = (features - features.mean()) / features.std(ddof=0)
    data = data.fillna(0)
    
    miss_enc = features.isnull().astype(int)
    miss_enc.columns = ["{}_mis".format(c) for c in miss_enc.columns]
    
    return pd.concat([data, miss_enc], axis=1, join='inner')
    
def main():
    env = kagglegym.make()
    obs = env.reset()
    
    # We use subset of training data, since TF on CPU is very slow
    train = obs.train[:30000]
    
    train_X = preprocess(train.ix[:, "derived_0":"technical_44"]).as_matrix().astype(np.float32)
    train_y = train.ix[:, "y"].as_matrix().astype(np.float32)
    

    n_features = train_X.shape[1]
    
    tx = tf.placeholder(tf.float32, (None, train_X.shape[1]))
    ty = tf.placeholder(tf.float32, (None, ))
    
    tW1 = tf.get_variable("W1", shape=(n_features, n_features))
    tW2 = tf.get_variable("W2", shape=(n_features, 1))
    
    tO1 = tf.nn.tanh(tf.matmul(tx, tW1))
    tO2 = tf.matmul(tO1, tW2)
    
    tMSE = tf.reduce_mean(tf.square(tf.subtract(tO2, train_y)))
    
    tOptimizer = tf.train.AdamOptimizer()
    tOptimize = tOptimizer.minimize(tMSE)
    
    batch_size = 32
    n_epochs = 1
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i_e in range(n_epochs):
            for i in range(0, train_X.shape[0], batch_size):    
                batch_X = train_X[i:i + batch_size, ...]
                batch_y = train_y[i:i + batch_size]
                
                _, loss = sess.run([tOptimize, tMSE], feed_dict={tx: batch_X, ty: batch_y})
                
                print(i, loss)
        
        while True:
            target = obs.target
            test_X = preprocess(obs.features.ix[:, "derived_0":"technical_44"]).as_matrix().astype(np.float32)
            
            y = sess.run(tO2, feed_dict={tx: test_X})
            target["y"] = y
                    
            obs, reward, done, info = env.step(target)
            
            if done:
                break
    
    print(info)
    
if __name__ == "__main__":
    main()