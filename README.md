# NLP
An implementation of the Transformer architecture + an mmaped GloVe vector store

# What it does right now
Nothing by itself. The forward-pass of the Transformer is implemented, but there's no CLI for it or a training loop (yet!). It's more or less a library implementation of arxiv.org/abs/1706.03762.
However, the GloVe vector store is pretty neat - it provides an mmapped kd-tree and (for now, a RAM) dictionary for GloVe vectors and tokens, respectively. 

# How to do it
Grab you some vectors from https://nlp.stanford.edu/projects/glove/ or something similarly formatted. Then, include Words.jl in your REPL and run 
```
G = GStore(path-to-glove.txt)
```
Careful - due to the mechanics of memory-mapping, the GloVe vectors and keys need to be copied into different files. Which means you might run out of disk. 
However, once this process is done, you can delete the original glove.txt file. 

# Usage
Once you have the various .bin files, you can index into G like so:

```
G["word"] = ...
```

or, 

```
vector = rand(300)

G[vector] = ...
```
