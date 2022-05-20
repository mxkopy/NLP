using FileIO, Pipe, DataStructures, Serialization, NearestNeighbors, StaticArrays

function parse_line( str )

    line = split(str, " ")
    return String( line[1] ), parse.(Float32, line[2:end])

end

function create_dict( filename )

    return SortedDict( readlines( filename ) .|> parse_line .|> ((word, vec),) -> word=>vec )

end


function serialize_glove_dict( input, output )

    dict = create_dict( input )
    serialize( output, dict )

end

# These functions create a nearest-neighbor search tree 
# which can be used in conjunction with a vector => word mapping 
# to retrieve natural language output from a model that outputs embeddings
function serialize_kd_tree( input, output )

    data::Vector{SVector{300, Float32}} = [ vector for (_, vector) in readlines( input ) .|> parse_line ]

    serialize( output, KDTree( data ) )

end

function embedding_from_word( dict, word )

    return dict[word]

end

function word_from_embedding( tree, vector, glovefile="data/glove.txt" )

    closest, _ = nn( tree, vector )

    line, _    = last( collect( zip(eachline(glovefile), 1:closest) ) )

    word, _    = parse_line( line )

    return word

end

function create_lookup_dictionaries(glovefile="data/glove.txt")

    serialize_glove_dict(glovefile, "data/dict")
    serialize_kd_tree(glovefile, "data/kd_tree")

end