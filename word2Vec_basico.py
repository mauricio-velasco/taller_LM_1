import torch
import torch.nn as nn
import torch.optim as optim
import random

#ACTIVIDAD 1: Un word2vec minimal

# 1. Generate dummy corpus
corpus = corpus = [
    "The cat sits on the mat.",
    "The dog sits on the mat.",
    "The cat chases the mouse.",
    "The dog chases the ball.",
    "A cat and a dog play together.",
    "The mat is under the cat.",
    "The ball rolls under the table.",
    "The mouse hides under the table.",
    "A car drives down the road.",
    "A tree stands by the road.",
    "The dog barks loudly at night.",
    "The cat sleeps on the warm roof.",
    "The mouse runs fast in the house.",
    "The ball bounces on the floor.",
    "The cat and dog eat food.",
    "The tree has green leaves.",
    "The car stops at the traffic light.",
    "The dog wags its tail happily.",
    "The cat climbs the tall tree.",
    "The mouse sneaks past the cat.",
    "The dog drinks water from the bowl.",
    "The cat watches the bird outside.",
    "The mouse nibbles cheese quietly.",
    "The ball is red and round.",
    "The tree grows tall and strong.",
    "The car honks loudly in traffic.",
    "The dog runs through the garden.",
    "The cat hides behind the curtain.",
    "The mouse escapes into the hole.",
    "The ball rolls across the grass.",
    "The tree provides shade on sunny days.",
    "The car speeds along the highway.",
    "The dog digs a hole in the yard.",
    "The cat purrs softly while resting.",
    "The mouse chews on the wire.",
    "The ball bounces off the wall.",
    "The tree's leaves fall in autumn.",
    "The car is parked near the house.",
    "The dog sleeps peacefully by the fire."
]


# 2. Preprocess: tokenize and build vocabulary
tokenized = [sentence.split() for sentence in corpus]
vocab = set(word for sentence in tokenized for word in sentence)
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

# 3. Generate training data for skip-gram: (center_word, context_word)
window_size = 2
data = []

for sentence in tokenized:
    indices = [word2idx[w] for w in sentence]
    for center_pos in range(len(indices)):
        center_word = indices[center_pos]
        for w_pos in range(max(0, center_pos - window_size), min(len(indices), center_pos + window_size + 1)):
            if w_pos != center_pos:
                context_word = indices[w_pos]
                data.append((center_word, context_word))

#EJERCICIO 1:
    #Cuente el numero de data pairs y verifique que su estimacion coincide con el conteo del codigo.
    #Escriba los primeros tres pares de entrenamiento en palabras.
    #Modifique el codigo de arriba y construya una ventana mayor.


print(f"Training data pairs: {len(data)}")

# 4. Define Skip-gram model
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, center_words):
        embeds = self.embeddings(center_words)      # (batch_size, embedding_dim)
        out = self.output(embeds)                   # (batch_size, vocab_size)
        return out

# 5. Training setup
vocab_size = len(vocab)
embedding_dim = 10
model = Word2Vec(vocab_size, embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 6. Training loop
num_epochs = 100
batch_size = 8

for epoch in range(num_epochs):
    total_loss = 0
    random.shuffle(data)
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        center_batch = torch.tensor([pair[0] for pair in batch])
        context_batch = torch.tensor([pair[1] for pair in batch])

        optimizer.zero_grad()
        output = model(center_batch)
        loss = criterion(output, context_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / (len(data) / batch_size)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# 7. Example: Show word embeddings for some words
for word in ["dog", "cat", "tree", "car"]:
    idx = torch.tensor([word2idx[word]])
    embed = model.embeddings(idx).detach().numpy()
    print(f"Embedding for '{word}': {embed}")

#Ejercicio 2: 
#   Haga un plot de la loss del modelo en el tiempo
#   Calcule las distancias angulares entre las 4 palabras del ejemplo anterior cuando el modelo esta entrenado
#   Como cambian sus resultados al modificar la dimension del embedding?


# ACTIVIDAD 2: Miramos un word2vec entrenado

import os
import gensim.downloader as api

# Download & load pretrained Word2Vec model (Google News, 300d)
model_name = 'word2vec-google-news-300'

print(f"Downloading and loading '{model_name}' model. This may take a few minutes...")

model = api.load(model_name)
api.BASE_DIR #Crea un archivo enorme aca que hay que eliminar despues...


# Now the model is ready to use:
print("Model loaded!")

# Example usage:
similar_words = model.most_similar('queen', topn=15)
print("Top 5 words similar to 'king':")
for word, score in similar_words:
    print(f"{word}: {score:.4f}")
#Example analogy (linear model hypothesis)
result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
print("king - man + woman ≈", result[0][0], f"(score: {result[0][1]:.4f})")



