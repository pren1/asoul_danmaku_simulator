import pickle
import pdb
from File_scan import File_scan
from tqdm import tqdm
import pandas as pd
import tf_glove
import numpy as np
import time

def process_single_data(single_file_path):
    'prepare training data'
    df = pd.read_pickle(single_file_path)
    df = df[df['message'].map(len) > 0]
    df.index = range(len(df))

    return df['message'].tolist()

if __name__ == '__main__':
    file_scan = File_scan("./Cleaned_database/")
    all_file_paths = file_scan.path_gen(extension='.pkl')
    all_documents = []
    for single_file_path in tqdm(all_file_paths):
        all_documents.extend(process_single_data(single_file_path))

    start_time = time.time()
    print('start training at', start_time)
    embedding_size = 256
    model = tf_glove.GloVeModel(embedding_size=embedding_size, context_size=500, min_occurrences=2000,
                                learning_rate=0.05, batch_size=4096)
    model.fit_to_corpus(all_documents)
    model.train(num_epochs=400)
    print('finish training, took', time.time() - start_time, 's')
    vocab = model.words

    corresponding_dict = model.get_word_to_id()

    with open('corresponding_dict.pickle', 'wb') as handle:
        pickle.dump(corresponding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    embeddings = model.embeddings
    print('vocab size', len(vocab))
    print('embedding shape', embeddings.shape)
    with open('embedding/glove-' + str(embedding_size) + '-words.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    np.save('embedding/glove-' + str(embedding_size), embeddings)

    res = np.load(f"embedding/glove-{embedding_size}.npy")
    model.generate_tsne('embedding_vectors.png', embeddings=res)