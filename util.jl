include("AnalogyFinder.jl")
using DelimitedFiles, PyCall; nltk = pyimport("nltk")

tokens_name = "data.glv"


# parts of speech tags, eye-read from nltk.upenn_tagset()
tags = ["\$", "(", ")", ",", "--", ".", "::", "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "PRP\$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP\$", "WRB", "''"]

# ------------------------------------------------PRE PROCESSING------------------------------------------------


# Returns a one-hot encoding that represents which part of speech (P.O.S.) the word is

function one_hot( tags::Vector{String}, pos::String )

    zeroes = zeros( length(tags) )

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

    zeroes = zeros( length(data), length( tags ) )

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
    

# Pads the clause up to num_states with the given string
function pad_clause( clause::Vector{String}; filler="..." )

    out = clause

    while( length(out) < num_states )

        push!( out, filler )

    end

    return lowercase.( out )

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

 

function write_data()

    open( tokens_name, "w" ) do io

        lines  = reduce( *, [ reduce(*, readlines( string( pwd(), "/books/", path ) ) ) for path in readdir("books") ])

        tokens = lines |> nltk.word_tokenize |> filter_clause 

        gloves = encode( tokens )

        pos_tags::Vector{String} = [ pos[2] for pos in nltk.pos_tag( tokens ) ]

        for ( token, i ) in zip( tokens, 1:length( tokens ))

            serialize( io, ( token, gloves[i, :], pos_tags[i] ) )

        end

    end

end



function next_clause( io, delims=[".", ";", "\"", "?", "!"] )

    vec_data::Matrix{ Float32 } = []

    one_hot::Matrix{ Int8 }     = []

    word, vec, onehot = deserialize( io )

    while !( word in delims )

        vec_data = cat(vec, vec_data; dims=1)
        one_hot  = cat(one_hot, onehot; dims=1)

        
    end

    return vec_data, one_hot

end



function find_longest_clause()

    m = 0

    open( tokens_name ) do io

        out = next_clause( io )

        while out != Nothing

            m   = max(m, length( out ) )

            out = next_clause( io )

        end

    end

    return m

end



write_data()