module Transformers
# https://arxiv.org/abs/1706.03762

export create_transformer, attention, head, multihead_attention, FFN_layer, MultiHead, Encoder, Decoder, Layer, Transformer

using Flux, SliceMap


function mask_function( length, exclude=[] )

    return collect( i in exclude ? 0 : 1 for i in 1:length )

end

nomask = Q -> i -> mask_function( size(Q)[1] )


# Q & K are assumed to be n x D matrices containing queries & keys
# assumes that the number of dimensions is by default the size of the second dimension
function attention( Q, K, V, mask, dims=size(Q)[2], sm=softmax )

    compatability = ( Q * transpose( K ) ./ sqrt( dims ) ) .* mask |> sm

    return compatability * V

end


function head( Q, K, V, qw, kw, vw, mask )

    return attention( Q * qw, K * kw, V * vw, mask )

end



function multihead_attention( Q, K, V, QW, KW, VW, WO, mask )

    out = map( zip(QW, KW, VW) ) do (qw, kw, vw)

        return head( Q, K, V, qw, kw, vw, mask )

    end

    return reduce( (l, r) -> cat(l, r, dims=2), out ) * WO

end


function FFN_layer( d_m )

    ffn = Chain( transpose, Dense(d_m, d_m), transpose )

    return SkipConnection( ffn, (fx, x) -> Flux.normalise( fx + x ) )

end



struct MultiHead

    QW::Vector{AbstractArray{<:Real}}
    KW::Vector{AbstractArray{<:Real}}
    VW::Vector{AbstractArray{<:Real}}
    WO::AbstractArray{<:Real}

end

MultiHead( d_m::Int, d_k::Int, d_v::Int, h::Int, init=Flux.glorot_normal ) = MultiHead( [init(d_m, d_k) for _ in 1:h], [init(d_m, d_k) for _ in 1:h], [init(d_m, d_v) for _ in 1:h], init( h * d_v, d_m ) ) 

function (layer::MultiHead)(Q, K, V, mask=1)

    out = multihead_attention( Q, K, V, layer.QW, layer.KW, layer.VW, layer.WO, mask )
    out = Flux.normalise( out + Q )

    return out
end

Flux.@functor MultiHead





struct Encoder

    multihead::MultiHead
    ffn

end

Encoder( d_m::Int, d_k::Int, d_v::Int, h::Int ) = Encoder( MultiHead(d_m, d_k, d_v, h), FFN_layer( d_m ) )

function (layer::Encoder)( X )

    out = layer.multihead(X, X, X)
    out = layer.ffn(out)

    return out
end

Flux.@functor Encoder





struct Decoder

    masked_multihead::MultiHead
    unmasked_multihead::MultiHead
    ffn

end

Decoder( d_m::Int, d_k::Int, d_v::Int, h::Int ) = Decoder( MultiHead(d_m, d_k, d_v, h), MultiHead(d_m, d_k, d_v, h), FFN_layer(d_m) )

function (layer::Decoder)(Q, K, V, mask)
    
    out = layer.unmasked_multihead(Q, Q, Q)
    out = layer.masked_multihead(out, K, V, mask)
    out = layer.ffn(out)

    return out
end

Flux.@functor Decoder




struct Layer

    encoder::Encoder
    decoder::Decoder

end

Layer( d_m::Int, d_k::Int, d_v::Int, h::Int ) = Layer( Encoder(d_m, d_k, d_v, h), Decoder(d_m, d_k, d_v, h) )

function (layer::Layer)( current_input, previous_output, mask )

    enc_out = layer.encoder( current_input )
    dec_out = layer.decoder( previous_output, enc_out, enc_out, mask )

    return enc_out, dec_out

end

Flux.@functor Layer


struct Transformer

    layers::Vector{Layer}
    start_token::AbstractArray
    end_token::AbstractArray

end

Transformer( d_m=300, d_k=50, d_v=50, h=8, depth=6; start_token=zeros(d_m) .- 1, end_token=ones(d_m) ) = Transformer( map( _ -> Layer(d_m, d_k, d_v, h), 1:depth ), start_token, end_token )

function (transformer::Transformer)(current_input, previous_output, mask=1)

    _, out = foldl( transformer.layers, init=(current_input, previous_output) ) do data, layer

        return layer(data..., mask)

    end

    out = Flux.mean( out, dims=1 )

    return cat( previous_output, out, dims=1 )

end

@Flux.functor Transformer ( layers,  )



end