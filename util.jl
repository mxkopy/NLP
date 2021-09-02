include("AnalogyFinder.jl")
using DelimitedFiles, PyCall; nltk = pyimport("nltk")

tokens_name = "data.glv"

# parts of speech tags, eye-read from nltk.upenn_tagset()
tags = ["\$", "(", ")", ",", "--", ".", "::", "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "PRP\$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP\$", "WRB", "''"]

delims=[".", ";", "\"", "?", "!"]

# ------------------------------------------------PRE PROCESSING------------------------------------------------


# Returns a one-hot encoding that represents which part of speech (P.O.S.) the word is

function one_hot( tags::Vector{String}, pos::String )

    zeroes = zeros( Int8, length(tags) )

    for i in 1:size( zeroes )[1]

        if tags[i] == pos

            zeroes[i] = 1

        else

            zeroes[i] = 0

        end

    end

    return zeroes

end

# Same as above but outputs a 2d one-hot matrix, for Vectors of data

function one_hot( tags::Vector{String}, data::Vector{String} )

    zeroes = zeros( Int8, length(data), length( tags ) )

    for i in 1:length( data )

        for k in 1:length( tags )

            if data[i] == tags[k]

                zeroes[ i, k ] = 1

            end

        end

    end

    return zeroes

end

# Filters for any unwanted words, like strange byte sequences or numbers
function filter_clause( clause; f=["\ufeff", string.(0:9)...] )

    return lowercase.( Vector{String}( filter!( x -> !(x in f), clause ) ) )

end
    

function serialize_glove()

    open("glove.glv", "w") do io

        for line in eachline("glove.txt", enc"UTF-8")

            word = split( line, ' ')[1]
            vec  = parse_line( line ) 

            serialize( io, ( word, vec ) )
    
        end

    end

end

function write_book( io, tokens )

    gloves = encode( tokens )

    pos_tags::Vector{String} = [ pos[2] for pos in nltk.pos_tag( tokens ) ]

    tags   = one_hot( tags, pos_tags )

    for ( token, i ) in zip( tokens, 1:length( tokens ))

        serialize( io, ( token, gloves[i, :], tags[i, :] ) )

    end

end

function write_lite()

    open( tokens_name, "w" ) do io

        for path in readdir("books")

            lines = reduce( *, readlines( string( pwd(), "/books/", path ) ) )

            tokens = lines |> nltk.word_tokenize |> filter_clause

            write_book( io, tokens )

        end

    end

end

test

function write_data()

    open( tokens_name, "w" ) do io

        lines  = reduce( *, [ reduce(*, readlines( string( pwd(), "/books/", path ) ) ) for path in readdir("books") ])

        tokens = lines |> nltk.word_tokenize |> filter_clause 

        write_book( io, tokens )

    end

end

# Pads the clause up to num_states with the given string
function pad_clause( word_vec::Matrix{Float32}, onehot_vec::Matrix{Int8}; filler="..." )

    onehot = one_hot( tags, "." )

    vec    = encode( "." )

    out_vec, out_onehot = word_vec, onehot_vec

    while ( size(out_vec)[2] < num_states )

        cat( out_vec, vec; dims=2 )
        cat( out_onehot, onehot; dims=2 )

    end

    return out_vec, out_onehot

end


function next_clause( io )

    # word, vec_data::Array{ Float32 }, one_hot::Array{ Int8 } = deserialize( io )
 
    word, vec_data::Array{ Float32 }, _onehot = deserialize( io )

    onehot  = one_hot( tags, _onehot )
    _one_hot = Array{Int8}( onehot )

    while !( word in delims )

        word, vec, _onehot = deserialize( io )

        onehot = one_hot( tags, _onehot )

        vec_data = cat( vec, vec_data; dims=2 )
        _one_hot  = cat( _one_hot, onehot; dims=2 )

        
    end

    return vec_data, _one_hot

end

function find_longest_clause()

    m = 0

    lines  = reduce( *, [ reduce(*, readlines( string( pwd(), "/books/", path ) ) ) for path in readdir("books") ])

    tokens = lines |> nltk.word_tokenize |> filter_clause 

    println("woa das fast")

    for token in tokens

        m = m + 1

        if token in delims

            m = 0

        end

    end

    return m

end

num_states  = find_longest_clause()

# write_data()