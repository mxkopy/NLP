include("AnalogyFinder.jl")
include("util.jl")

using Flux, PyCall, Serialization, DataStructures; nltk = pyimport("nltk")

# Attention-based NLP generator
# Given one sentence, tries to guess the next one. 

# number of dimensions that glove vecs gave
glove_dims  = 300

num_states  = find_longest_clause()
num_latent  = 2 * num_states

latent_size = glove_dims + length( tags )

head_size  = 3

# inspired by VAE's, we use dense layers to learn std and mean of a distribution
sentim_enc   = Chain( Dense( glove_dims, latent_size ) )
struct_enc   = Chain( Dense( length( tags ), latent_size ) )

# reconstructs features from the latent space
sentim_dec   = Dense( latent_size, glove_dims )
struct_dec   = Dense( latent_size, length( tags ) )

attn         = Chain( Dropout( 0.7 ), Dense( num_latent, num_states ), softmax, transpose, Dense( latent_size, latent_size ) )




# Called right before input to the model
function process_clause( clause::Vector{String} )

    clause    = filter_clause( clause )

    clause    = pad_clause( clause )

    pos_tags  = nltk.pos_tag( clause )

    # using AnalogyFinder.jl, encodes the vector of strings by looking up glove vectors, usually in "glove.txt"

    gloves    = encode( clause )

    # nltk.pos_tag outputs a tuple of the word and its corresponding part of speech, so we select only for it

    hot_zeros = one_hot( tags, collect( [ tag[2] for tag in pos_tags ]  ) )

    return transpose( gloves ) , transpose( hot_zeros )

end


function attention( layer, latent, prev_latent )

    _latent    = cat( latent, prev_latent; dims=2 )

    return layer( transpose( _latent )  )

end

# given the model, a one-hot struct vector, and a real-valued glove vector
# outputs the autoencoded sentiment and glove embedding 

# the struct vector output should be softmaxed, and should be interpreted as a one-hot (using the maximum)
function forward( model, prev_latent, glove_vec, one_hot, param )

    sentim_enc, struct_enc, sentim_dec, struct_dec, attn = model

    # Encoding process
    std        = sentim_enc( glove_vec )
    mean       = struct_enc( one_hot )

    # Introduction of noise and forming the latent space
    _latent   = mean .+ ( std .* param )

    # featurewise and statewise attention
    latent     = attention( attn, _latent, prev_latent )

    # decoding of the   next data given the current context
    next_sentim = sentim_dec( latent ); 
    next_struct = struct_dec( latent );

    # decoding of the   current data given the previous context
    sentiment = sentim_dec( prev_latent )
    structure = struct_dec( prev_latent )

    return latent, sentiment, structure, next_sentim, next_struct

end



function train_model( model )

    opt         = ADAM( 0.01 ); 

    r_loss_auto, r_loss_next, l_loss_auto, l_loss_next, count = 0.0, 0.0, 0.0, 0.0, 0

    std_enc, mean_enc, sentim_dec, struct_dec, attention = model

    parameters  = Flux.params( std_enc, mean_enc, sentim_dec, struct_dec, attention  )

    latent = Flux.glorot_uniform( glove_dims + length( tags ), num_latent - num_states )

    open( tokens_name ) do io
        
        # Target sentiment and structure data
        sentiment_t, structure_t      = io |> next_clause |> pad_clause
        sentiment_t_n, structure_t_n  = io |> next_clause |> pad_clause

        # Training loop

        while !eof( io )

            rands  = ones( latent_size )

            gs = gradient( parameters ) do

                latent, sentiment, structure, next_sent, next_struct = forward( model, latent, sentiment_t, structure_t, rands )

                r_loss_auto = Flux.Losses.mse( sentiment, sentiment_t )
                l_loss_auto = Flux.Losses.logitbinarycrossentropy( structure, structure_t )

                r_loss_next = Flux.Losses.mse( next_sent, sentiment_t_n )
                l_loss_next = Flux.Losses.logitbinarycrossentropy( next_struct, structure_t_n )

                return ( r_loss_auto + l_loss_auto ) + 0.1 * ( r_loss_next + l_loss_next )

            end

            count = count + 1

            if count == 100

                serialize( "model.bson", ( model, latent ) )
                count = 0

            end
        
            Flux.Optimise.update!(opt, parameters, gs)

            sentiment_t, structure_t      = io |> next_clause |> pad_clause |> process_clause
            sentiment_t_n, structure_t_n  = io |> next_clause |> pad_clause |> process_clause
    
            println( '\r', r_loss_auto , ' ', l_loss_auto, " | ", r_loss_next, ' ',  l_loss_next )
    
        end

    end


end

# Evaluates the model on a single clause.
function eval_model( model, latent, data )

    model = testmode!( model )

    _sentiment, _structure = nltk.word_tokenize(data) |> pad_clause |> process_clause

    latent, sentiment, structure, next_sent, next_struct = forward( model, latent, _sentiment, _structure, ones( latent_size ) )

    sentiment = transpose( sentiment ) 
    next_sent = transpose( next_sent )

    println( decode(sentiment) )
    println( decode(next_sent) )

end

# model, latent = deserialize("model.bson")

# eval_model( model, latent, "The gentleman pronounced him to be a fine figure of a man.")

train_model( (sentim_enc, struct_enc, sentim_dec, struct_dec, attn) )

# open("flat.txt") do io

#     test      = io |> next_clause |> pad_clause

#     print(test)

#     data      = test |> process_clause  

#     for i in 1:length( size(data[1]) )

#         if data[1][i][1] == 0.0

#             println(i)

#         end

#     end

# end