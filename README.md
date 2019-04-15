# Text_Summarization
IPython Notebook
Go to (demo/Legal_Doc_Summarizer.ipynb)
Data
text -- (demo/data/text2.pkl)
summary -- (demo/data/summary2.pkl)
keras-text-summarization
Text summarization using seq2seq and encoder-decoder recurrent networks in Keras

Machine Learning Models
There are currently 3 other encoder-decoder recurrent models based on some recommendation here

The implementation can be found in keras_text_summarization/library/rnn.py

The One used for training
Recursive RNN 1 (RecursiveRNN1 in rnn.py): The recursive RNN 1 takes the artcile content and the current built-up summarized text to predict the next character of the summarized text.
training: run demo/recursive_rnn_v1_train_legal.py
prediction: run demo/recursive_rnn_v1_predict_legal.py
The trained model is available in the demo/models folder

Output on Full Data
Legal_gen_summary.txt
