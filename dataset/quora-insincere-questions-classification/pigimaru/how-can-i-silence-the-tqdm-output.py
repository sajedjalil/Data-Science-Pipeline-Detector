import torchtext

print('Start')
embedding_data_path = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
tmp_vectors=torchtext.vocab.Vectors(embedding_data_path, max_vectors=None)
print('Finish')