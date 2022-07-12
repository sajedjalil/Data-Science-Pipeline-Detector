__import__('pandas').read_csv('../input/deepfake-detection-challenge/sample_submission.csv',converters={'label':lambda e:.5}).to_csv('submission.csv',index=False)
