from __future__ import print_function

import pandas as pd
from keras_text_summarization.library.rnn import RecursiveRNN1
import numpy as np
import pickle

def main():
    np.random.seed(42)
    data_dir_path = './data'
    model_dir_path = './models'

    print('loading csv file ...')
    
    with open(data_dir_path + '/summary2.pkl', 'rb') as f:
        list_of_summaries = pickle.load(f)
    with open(data_dir_path + '/text2.pkl', 'rb') as f:
        list_of_text = pickle.load(f)
    
    # df = df.loc[df.index < 1000]
    X = list_of_text
    Y = list_of_summaries

    config = np.load(RecursiveRNN1.get_config_file_path(model_dir_path=model_dir_path)).item()

    summarizer = RecursiveRNN1(config)
    summarizer.load_weights(weight_file_path=RecursiveRNN1.get_weight_file_path(model_dir_path=model_dir_path))

    print('start predicting ...')
    orig_summary = []
    generated_summary = []
    for i in np.random.permutation(np.arange(len(X)))[0:20]:
        x = X[i]
        actual_summary = Y[i]
        orig_summary.append(actual_summary)
        
        gen_summary = summarizer.summarize(x)
        generated_summary.append(gen_summary)
        
        # print('Article: ', x)
        print('--Generated Summary: ', gen_summary)
        print('--Original Summary: ', actual_summary)
    
    with open('gen_summary.pkl', 'wb') as f:
        pickle.dump(generated_summary, f)
    with open('orig_summary.pkl', 'wb') as f:
        pickle.dump(orig_summary, f)


if __name__ == '__main__':
    main()
