import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import streamlit as st
import requests
import nltk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NextChar(nn.Module):
  def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, emb_dim)
    self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
    self.lin2 = nn.Linear(hidden_size, vocab_size)

  def forward(self, x):
    x = self.emb(x)
    x = x.view(x.shape[0], -1)
    x = torch.sin(self.lin1(x))
    x = self.lin2(x)
    return x
    

# Function to fetch text content from URL
def fetch_text(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        st.error(f"Failed to fetch the file. Status code: {response.status_code}")

# Function to tokenize text into characters
def tokenize_text(text_content):
    words = text_content.split()
    chars = sorted(list(set(''.join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi[' '] = 0
    itos = {i: s for s, i in stoi.items()}
    return words, chars, stoi, itos

# Function to tokenize sentences into characters
def tokenize_sentences(sentences, stoi):
    block_size = st.sidebar.slider("Block Size", min_value=1, max_value=100, value=50, step=1)
    X, Y = [], []
    for sentence in sentences:
        chars = list(sentence)
        context = [0] * block_size
        for ch in chars + ['.']:
            ix = stoi.get(ch, 0)
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)

# Function to create the NextChar model
def create_model(block_size, vocab_size, emb_dim, hidden_size):
    model = NextChar(block_size, vocab_size, emb_dim, hidden_size)
    return model.to(device)

# Function to generate names from the trained model
def generate_name(context, model, itos, stoi, block_size, k):
    context = [0] * block_size
    name = ''
    for i in range(k):
        x = torch.tensor(context).view(1, -1).to(device)
        y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        ch = itos[ix]
        if ch == '.':
            break
        name += ch
        context = context[1:] + [ix]
    return name

# Streamlit app
def main():
    st.title("Character Prediction Streamlit App")
    
    # User input for random seed
    random_seed = st.sidebar.number_input("Random Seed", value=4000002)
    
    # User input for embedding size
    emb_dim = st.sidebar.selectbox("Embedding Size", [4, 8, 12, 16], index=1)
    
    # Fetching text content
    text_content = fetch_text("https://cs.stanford.edu/people/karpathy/char-rnn/shakespear.txt")
    words, chars, stoi, itos = tokenize_text(text_content)
    
    # Tokenizing sentences
    sentences = nltk.sent_tokenize(text_content)
    X, Y = tokenize_sentences(sentences, stoi)
    
    # Model parameters
    vocab_size = len(stoi)
    hidden_size = 10
    
    # Create and train model
    if st.button("Train Model"):
        model = create_model(X.shape[1], vocab_size, emb_dim, hidden_size)
        
        # Training
        loss_fn = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        batch_size = 4096
        for epoch in range(1000):
            for i in range(0, X.shape[0], batch_size):
                x = X[i:i + batch_size].to(device)
                y = Y[i:i + batch_size].to(device)
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                loss.backward()
                opt.step()
                opt.zero_grad()
        
        st.success("Model training completed!")
        
        # User input for context and k
        context = st.text_input("Enter Context")
        k = st.sidebar.number_input("Number of Characters to Predict (k)", value=10)
        
        # Generate and display names
        if context:
            generated_name = generate_name(context, model, itos, stoi, X.shape[1], k)
            st.write("Generated Name:", generated_name)

if __name__ == "__main__":
    main()
