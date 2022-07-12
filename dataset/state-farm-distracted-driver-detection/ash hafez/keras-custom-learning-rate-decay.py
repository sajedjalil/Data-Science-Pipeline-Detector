import numpy as np 
import keras

# ----
# NOTE: Needs latest version of Keras for the keras.callbacks module
# ----
'''
class LearningRateDecay(keras.callbacks.Callback):

    # ----------------------------------------------------------------------------
    
    def __init__(self, decay_rate=0.5, patience=3, weights_path=None, improve_delta=1e-4):

        super(LearningRateDecay, self).__init__()

        self.decay_rate     = np.float32( decay_rate )
        self.patience       = patience
        self.weights_path   = weights_path
        self.best_score     = np.Inf
        self.fail_count     = 0
        self.improve_delta  = improve_delta

    # ----------------------------------------------------------------------------
    
    def on_train_begin(self, logs={}):

        self.best_lr    = np.float32( self.model.optimizer.lr.get_value() )

        self.model.save_weights( self.weights_path, overwrite=True )

    # ----------------------------------------------------------------------------
    
    def on_epoch_begin(self, epoch, logs={}):
        pass

    # ----------------------------------------------------------------------------
    
    def on_batch_begin(self, batch, logs={}):
        pass

    # ----------------------------------------------------------------------------
    
    def on_batch_end(self, batch, logs={}):
        pass

    # ----------------------------------------------------------------------------
    
    def on_epoch_end(self, epoch, logs={}):
        
        score   = logs.get('val_loss')

        print("")          
        
        if score < (self.best_score - self.improve_delta): 

            self.fail_count  = 0
            self.best_score  = score
            self.best_lr     = np.float32( self.model.optimizer.lr.get_value() )
            
            self.model.save_weights( self.weights_path, overwrite=True )
        
        else:
    
            self.fail_count  += 1
                        
            self.model.load_weights( self.weights_path )          

            if self.fail_count >= self.patience:

                self.model.stop_training = True

            else:
  
                print( ">> Reducing lr [%.12f -> %.12f]" % ( self.best_lr, self.best_lr * self.decay_rate) )                                      

                self.best_lr    = self.best_lr * self.decay_rate
                
                self.model.optimizer.lr.set_value( self.best_lr )

        print( "Epoch[%2d] Fails[%2d] | Score[%.6f] | Best[%.6f]" % (epoch+1, self.fail_count, score, self.best_score ) )

    # ----------------------------------------------------------------------------
    
    def on_train_end(self, logs={}):
        pass
'''
# ----

print("Done")

