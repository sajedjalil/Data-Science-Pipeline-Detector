__import__('pandas').read_csv('../input/sample_submission.csv',converters={'EncodedPixels':lambda e:''}).to_csv('submission.csv',index=False)
