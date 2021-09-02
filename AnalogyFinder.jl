using Pipe, StringEncodings, CodecZstd, CodecZlib, TranscodingStreams, ZfpCompression, Serialization, DataStructures

global distance = ( w1, w2 ) -> ( w1 .- w2 ) .^ 2 |> sum

global glovedir = "glove.txt"

# converts a string of floats into an actual array of floats
function parse_line( line )

    return @pipe line |> split( _, ' ')[ 2 : end ] |> parse.( Float32, _ )

end


function encode( words::Vector{String}; glove_dims=300 )

    sorted_indices::Vector{Int}  = sortperm( words )

    output                       = zeros( length( words ), glove_dims )

    num = 0

    open("glove.glv") do io

        while !eof( io )

            glove_word, vec  = deserialize( io )

            removed::Vector{Int} = [] 

            for index in sorted_indices

                if words[ index ] == glove_word

                    output[ index, : ] = vec
                    removed = push!( removed, index )

                end

            end

            sorted_indices = filter!( x -> !(x in removed), sorted_indices )

            num = num + 1

            if length( sorted_indices ) == 0

                return output

            end

            print('\r', num / 1917494.0, ' ', length( sorted_indices ) )

        end

    end

    return output

end


# retrieves the glove embeddings of a word
function encode( word::String; filler="..." )

    count = 0

    open("glove.glv") do io

        while !eof(io)

            _word, vec = deserialize( io )

            if word == _word

                return vec

            end

            count = count + 1

        end

    end

    return zeros( 300 )

end



function decode( _glove_vec::Vector{Float64} )

    minimum = 100.0; word = ""

    open("glove.glv", "r") do io

        while !eof( io )

            glove_str, glove_vec = deserialize( io )

            _min = reduce( +, abs.( glove_vec .- _glove_vec ) )

            if _min < minimum

                minimum = _min

                word    = glove_str

            end

        end

    end

    return word

end

# TODO:

# You can do binary lexographic search on glove.txt iff its sorted s.t. string(x) < string(y) => x < y

function decode( glove_vectors )

    mins = zeros( size( glove_vectors )[1] ) .+ 200.0; words = Vector{String}( undef, size( glove_vectors )[1] )

    count = 0

    open("glove.glv", "r") do io

        while !eof(io)

            word, glove_vec = deserialize( io )

            for i in 1:size( glove_vectors )[1]

                diff = reduce(+, abs.( glove_vectors[i, :] .- glove_vec) )

                if diff < mins[i]

                    mins[i]  = diff
                    words[i] = word

                end

            end

            count = count + 1

            print('\r', count / 1917494.0 )

        end

    end

    return words

end



function word_diff( word1, word2 )

    w1, w2 = encode( word1 ), encode( word2 )

    if w1 == Nothing || w2 == Nothing

        return Nothing

    end

    return distance( w1, w2 )

end
