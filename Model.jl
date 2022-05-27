using Flux


function mask( length, exclude=[] )

    return collect( i in exclude ? 0 : 1 for i in 1:length )

end



function attention( Q, K, V, mask )

    compatability = ( Q * transpose( K ) ./ sqrt( size(Q)[2] ) ) .* mask |> softmax

    return compatability * V

end



function head( Q, K, V, qw, kw, vw, mask )

    return attention( Q * qw, K * kw, V * vw, mask )

end


# m, in this case, is a function that takes in the current position of the word being attended to 
# so for regular multihead attention we just set it to a default-mask (i.e. all ones)
# for masked multiheaded attention we can define m to be the mask of everything but the current position
function multihead_attention( Q, K, V, QW, KW, VW, WO, m=_ -> mask( size(Q)[1] ) )

    out = map( 1:length(QW) ) do i

        return head( Q, K, V, QW[i], KW[i], VW[i], m(i) )

    end

    return cat( out..., dims=2 ) * WO

end



function normalize( matrix, dims=2:ndims(matrix) )

    return mapslices( matrix, dims=collect(dims)) do vector

        min = minimum( vector )
        max = maximum( vector )

        return 2 * ( (vector .- min) / ( max - min ) ) .- 1

    end

end



function multihead_layer( d_m, d_k, d_v, h, m=Q -> _ -> mask( size(Q)[1] ) )

    QW = [ rand(d_m, d_k) for _ in 1:h ]
    KW = [ rand(d_m, d_k) for _ in 1:h ]
    VW = [ rand(d_m, d_v) for _ in 1:h ]
    WO = rand( h * d_v, d_m )

    attend = (Q, K, V) -> multihead_attention( Q, K, V, QW, KW, VW, WO, m(Q) )

    layer  = (Q, K, V) -> normalize( attend( Q, K, V ) + Q )

    return layer, (QW, KW, VW, WO)

end



function FFN_layer( d_m )

    dense = Dense( d_m, d_m )

    FFN   = x -> x |> transpose |> dense |> transpose 

    layer = input -> normalize( FFN( input ) + input )

    return layer, dense

end



function create_encoder( d_m=300, d_k=50, d_v=50, h=8 )

    multihead, params = multihead_layer(d_m, d_k, d_v, h)

    ffn, dense        = FFN_layer( d_m )

    return input -> ffn( multihead( input, input, input ) ), (params..., dense)

end



function create_decoder( d_m=300, d_k=50, d_v=50, h=8 )
    
    m = Q -> i -> mask( size(Q)[1], collect( i:size(Q)[1] ) )

    masked_multihead, params_1 = multihead_layer( d_m, d_k, d_v, h, m )

    multihead, params_2        = multihead_layer( d_m, d_k, d_v, h )

    ffn, dense                 = FFN_layer( d_m )

    decoder_I                  = input -> masked_multihead(input, input, input)

    decoder_II                 = (Q, K, V) -> ffn( multihead(Q, K, V) )

    return (Q, K, V) -> decoder_II( decoder_I( Q ), K, V ), (params_1..., params_2..., dense)

end

function layer_forward( encoder, decoder, input, output )

    enc_out = encoder( input )
    dec_out = decoder( output, enc_out, enc_out )

    return enc_out, dec_out

end

function create_layer( d_m=300, d_k=50, d_v=50, h=8 )

    encoder, enc_params = create_encoder( d_m, d_k, d_v, h )
    decoder, dec_params = create_decoder( d_m, d_k, d_v, h )

    layer = (input, output) -> layer_forward( encoder, decoder, input, output )

    return layer, (enc_params..., dec_params...)

end


function create_stack( d_m=300, d_k=50, d_v=50, h=8, depth=6 )

    layers_params = [ create_layer(d_m, d_k, d_v, h) for _ in 1:depth ]

    layers = [ layer for (layer, _) in layers_params ]
    params = [ param for (_, param) in layers_params ]

    stack  = foldl( layers ) do cumulative, layer

        return (i, o) -> layer( cumulative(i, o)... )

    end

    return stack, params

end




# encoder, params = create_encoder(300, 50, 50, 8)

# words = randn(10, 300)

