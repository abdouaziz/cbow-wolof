import streamlit  as st
from cbow import gradient_descent
from cbow import get_dict
import re
import nltk
from matplotlib import pyplot 
from utils import compute_pca
import time 
import os 
nltk.download('punkt')


st.title("CBOW : WOLOF - FRENCH- ENGLISH")  

 
def file_selector(folder_path='./data/'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

filename = file_selector()
st.write('You selected `%s`' % filename)

@st.cache
def load_data(filename=filename) :
    with open(file=filename ,  encoding='utf8') as f :
        data  = f.read()
        data = re.sub(r'[,!?;-]', '.',data)                                  
        data = nltk.word_tokenize(data)                                     
        data = [ ch.lower() for ch in data if ch.isalpha() or ch == '.']  

    return data 


data = load_data()
word2Ind, Ind2word = get_dict(data)


C = 2
N = 50
V = len(word2Ind)
num_iters = 150

W1, W2, b1, b2 = gradient_descent(data, word2Ind, N, V, num_iters) 

latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    # Update the progress bar with each iteration.
    latest_iteration.text(f'Iteration {i+1}')
    bar.progress(i + 1)
time.sleep(0.1)
words = load_data()[:100]

embs = (W1.T + W2)/2.0
idx = [word2Ind[word] for word in words]
X = embs[idx, :]
 
result= compute_pca(X, 2)  

pyplot.scatter(result[:,0], result[:,1])
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i,0], result[i,1]))
st.pyplot() 



 