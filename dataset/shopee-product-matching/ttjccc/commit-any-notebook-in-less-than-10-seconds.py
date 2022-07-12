# %% [code] {"jupyter":{"outputs_hidden":false}}
import pandas as pd 
test = pd.read_csv('../input/shopee-product-matching/test.csv')
sample_submission = pd.read_csv('../input/shopee-product-matching/sample_submission.csv')
if len(test)==3:
    sample_submission.to_csv('submission.csv')
    import sys
    sys.exit()
# If you want to change some hyper-parameters and make another submission quickly:
# Add the above code to the top of your notebook, then go to `File` -> `Editor Type` and select `Script` before committing.

# explanation: in notebook mode, execption from one cell does not stop the exectution for the next cells,
# while in script mode, you can stop the script early programmatically.
# For more information, see: https://www.kaggle.com/code-competition-debugging