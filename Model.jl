using Flux

# This is an attention-based natural language processor. 
# I truly think autoencoders are the most logical starting point for any learning task. 
# The design specifications for autoencoders are very straightforward; namely 


# The input is a string of fixed length; the output is the same string. 

# The output of each layer gets progressively narrower; except for the decoder, which undoes this narrowing. 


# THE ATTENTION MECHANISM 

# Attention is achieved through 'sliding windows' 