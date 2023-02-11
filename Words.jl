using FileIO, Pipe, DataStructures, Serialization, NearestNeighbors, StaticArrays, Distributed, Mmap, OrderedCollections

import Base: getindex

@everywhere function parse_line( str::String; precision=Float32 )

    line = split(str, " ") 

    dims = length(line) - 1

    key, value::SVector{dims, precision} = String(line[1]), parse.(precision, line[2:end])

    return key => value

end

function split_glove(; glove="glove.txt", values_out="glove.values.bin" )

    d = glove |> eachline |> first |> parse_line |> last |> length |> Int64
    n = parse(Int64, read(`bash -c "wc -l < $glove"`, String))

    open(values_out, "a+") do V

        write(V, d)
        write(V, n)

        for kv in Iterators.map( parse_line, eachline(glove) )

            write(V, kv.second)
            
        end

    end

end

function create_glove_lookup_dictionary( glove::String="glove.txt" )

    return Iterators.map( glove |> eachline |> enumerate ) do (i, line)

        return (line |> parse_line |> first) => i 

    end |> OrderedDict

end

function mmap_values( values::String="glove.values.bin" )

    V = open(values, "r")

    d = read(V, Int64)
    n = read(V, Int64)

    close(V)

    return mmap(values, Matrix{Float32}, (d, n), sizeof(d) + sizeof(n))

end

struct GStore

    dictionary
    tree

end

function GStore(; values="glove.values.bin", dictionary="dictionary.bin", tree="kd_tree.bin" )

    D = deserialize(dictionary)
    T = injectdata(deserialize(tree), mmap_values(values))

    return GStore(D, T)

end

function GStore( glove::String )

    split_glove( glove=glove )

    serialize("dictionary.bin", create_glove_lookup_dictionary( glove ))
    serialize("kd_tree.bin", DataFreeTree(KDTree, mmap_values() )) 

    return GStore()

end



getindex( gstore::GStore, key::String ) = getindex( G.tree.data, getindex( gstore.dictionary, key ) ) |> collect

getindex( gstore::GStore, key::Vector{T} ) where T <: Number = Iterators.drop( gstore.dictionary |> keys, first( nn(gstore.tree, key) ) - 1 ) |> first

getindex( gstore::GStore, keys::Vararg{Vector{T}} ) where T <: Number = map( key -> getindex(gstore, key), keys )
