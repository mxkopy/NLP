# https://arxiv.org/abs/1706.03762

using Flux, SliceMap, CUDA, Zygote

nrow(x::AbstractVector) = length(x)

nrow(x::AbstractMatrix) = size(x)[1]
ncol(x::AbstractMatrix) = size(x)[2]

# ----------------------------------------------------
# non-stateful functions used to build everything else
# ----------------------------------------------------

function mask(A)

    Zygote.ignore() do 

        for i in 1:nrow(A), j in 1:ncol(A)

            A[i, j, :] .= j <= i ? A[i, j, :] : -Inf

        end

    end

    return A

end


function attention( Q, K, V, masked, dims=nrow(Q) )

    compatability = transpose(K) * Q ./ sqrt(dims)
    compatability = softmax( masked ? mask(compatability) : compatability, dims=2 )

    return V * compatability

end



function head( Q, K, V, qw, kw, vw, masked )

    return attention( qw * Q, kw * K, vw * V, masked )

end



function multihead_attention( Q, K, V, QW, KW, VW, WO, masked )

    out = mapreduce( (l, r) -> cat(l, r, dims=1), zip(QW, KW, VW) ) do (qw, kw, vw)

        return head(Q, K, V, qw, kw, vw, masked)

    end

    return WO * out

end


# ---------------------------------------
# stateful bits with trainable parameters
# ---------------------------------------

struct MultiHead

    QW::Vector{AbstractArray{<:Real}}
    KW::Vector{AbstractArray{<:Real}}
    VW::Vector{AbstractArray{<:Real}}
    WO::AbstractArray{<:Real}
    LN::LayerNorm

end

MultiHead( d_m::Int, d_k::Int, d_v::Int, h::Int, init=Flux.glorot_normal ) = MultiHead( [init(d_k, d_m) for _ in 1:h], [init(d_k, d_m) for _ in 1:h], [init(d_v, d_m) for _ in 1:h], init( d_m, h * d_v ), LayerNorm(d_m) ) 

function (layer::MultiHead)(Q, K, V, masked=false)

    out = multihead_attention( Q, K, V, layer.QW, layer.KW, layer.VW, layer.WO, masked )
    out = layer.LN(out + Q)

    return out

end

Flux.@functor MultiHead





struct FFN

    ffn
    LN::LayerNorm

end

FFN( d_m::Int ) = FFN( Dense(d_m, d_m), LayerNorm(d_m) )

function (layer::FFN)( X )

    out = layer.ffn(X)
    out = layer.LN(out + X)

    return out

end

Flux.@functor FFN





struct Encoder

    multihead::MultiHead
    ffn::FFN

end

Encoder( d_m::Int, d_k::Int, d_v::Int, h::Int ) = Encoder( MultiHead(d_m, d_k, d_v, h), FFN(d_m) )

function (layer::Encoder)( X )

    out = layer.multihead(X, X, X)
    out = layer.ffn(out)

    return out
end

Flux.@functor Encoder





struct Decoder

    masked_multihead::MultiHead
    unmasked_multihead::MultiHead
    ffn::FFN

end

Decoder( d_m::Int, d_k::Int, d_v::Int, h::Int ) = Decoder( MultiHead(d_m, d_k, d_v, h), MultiHead(d_m, d_k, d_v, h), FFN(d_m) )

function (layer::Decoder)(Q, K, V)
    
    out = layer.unmasked_multihead(Q, Q, Q)
    out = layer.masked_multihead(out, K, V, true)
    out = layer.ffn(out)

    return out
end

Flux.@functor Decoder





struct Layer

    encoder::Encoder
    decoder::Decoder

end

Layer( d_m::Int, d_k::Int, d_v::Int, h::Int ) = Layer( Encoder(d_m, d_k, d_v, h), Decoder(d_m, d_k, d_v, h) )

function (layer::Layer)( inputs, outputs )

    enc_out = layer.encoder( inputs )
    dec_out = layer.decoder( outputs, enc_out, enc_out )

    return enc_out, dec_out

end

Flux.@functor Layer



# Inputs are assumed to be (d_m, n) size 

struct Transformer

    layers::Vector{Layer}
    start_token::AbstractArray
    end_token::AbstractArray

end

Transformer( d_m=64, d_k=8, d_v=8, h=8, depth=6; start_token=zeros(d_m, 1) .- 1, end_token=ones(d_m, 1) ) = Transformer( map( _ -> Layer(d_m, d_k, d_v, h), 1:depth ), start_token, end_token )

function (transformer::Transformer)(inputs::AbstractArray, outputs::AbstractArray)

    enc, dec = foldr( transformer.layers, init=(inputs, outputs) ) do layer, data

        return layer( data... )

    end

    return getindex( dec, Flux.argmax( dec, dims=2 ) )

end

@Flux.functor Transformer (layers, )

Flux.gpu(x::Transformer) = Transformer( x.layers |> gpu, x.start_token |> gpu, x.end_token |> gpu )
Flux.cpu(x::Transformer) = Transformer( x.layers |> cpu, x.start_token |> cpu, x.end_token |> cpu )



function encode_position(pos::Int, d, max)

    return map(1:d) do i

        w = max ^ ( ((i ÷ 2) * 2) / d )

        isodd(i) ? sin( pos / w ) : cos( pos / w )

    end

end

encode_position(pos::Tuple{Int}, d, max) = encode_position(pos..., d, max)

function encode_position(pos::Tuple{Int, Int}, d, max)

    return vcat( encode_position( first(pos), d÷2, max ), encode_position( last(pos), d÷2, max ) )

end

isend(transformer::Transformer, token::AbstractVecOrMat) = isequal(transformer.end_token, token)


struct TransformerIterator

    transformer::Transformer
    inputs::AbstractArray
    bound::Number

    TransformerIterator(T, I, B=Inf) = new( T, cat( I, T.end_token, dims=2), B )

end

Base.length(itr::TransformerIterator) = itr.bound

function Base.iterate( T::TransformerIterator, outputs=T.transformer.start_token )

    out     = T.transformer(T.inputs, outputs)
    outputs = cat(outputs, out, dims=2)

    return ncol(outputs) > T.bound + 1 || isend(T.transformer, out) ? nothing : (outputs[:, 2:end], outputs)

end


function (transformer::Transformer)( inputs::AbstractArray, bound::Number=reduce(*, size(inputs)[2:end]) )

    inputs = mapreduce(hcat, Iterators.product( axes(inputs)[2:end]... )) do pos

        return reshape(inputs[:, pos...], :) + encode_position(pos, nrow(transformer.start_token), 2048)

    end

    return TransformerIterator(transformer, inputs, bound)

end