module InceptionVAE


using Flux, Serialization, WAV, Zygote, Distributions, CUDA


function conv_block( conv_type, kernel, in_channels, out_channels, stride=1 )

    return Chain(
        
        conv_type( kernel, in_channels => out_channels, pad=SamePad(), stride=stride ),
        BatchNorm( out_channels ),
        x -> relu.(x)
    )

end



function conv_group( conv_type, channels, bottleneck_channels, connection )

    bn      = conv_block( conv_type, (1, 1), channels, bottleneck_channels )

    conv1   = conv_block( conv_type, (1, 1), bottleneck_channels, channels )
    conv3   = conv_block( conv_type, (3, 3), bottleneck_channels, channels)

    conv5   = conv_block( conv_type, (3, 3), bottleneck_channels, bottleneck_channels )
    conv5v  = conv_block( conv_type, (3, 1), bottleneck_channels, channels )
    conv5h  = conv_block( conv_type, (1, 3), bottleneck_channels, channels )

    pool3   = MaxPool( (3, 3), stride=1, pad=1)
    conv1_p = conv_block( conv_type, (1, 1), bottleneck_channels, channels )

    g1    = Parallel( connection, conv5h, conv5v )
    c1    = Chain( conv5, g1 )

    g2    = Parallel( connection, conv1, conv3, c1, conv1_p ∘ pool3 )
    c2    = Chain( bn, g2 )

    return c2

end


function encoder_module( channels )

    # connection in the paper is (x...) -> cat( x..., dims=3), but that is super resource intensive
    return conv_group( Conv, channels, channels ÷ 2, + )

end



function decoder_module( channels )

    return conv_group( ConvTranspose, channels, channels ÷ 4, + )

end



function downsampler( kernel, in_channels, out_channels )

    return Chain( 

        MeanPool( ( kernel, kernel) ), 
        Conv( (1, 1), in_channels => out_channels ),
        BatchNorm( out_channels ), 
        x -> relu.(x)
    )

end



function upsampler( upsample, in_channels, out_channels )

    return Chain( 

        Upsample(upsample),
        ConvTranspose( (3, 3), in_channels => out_channels, pad=SamePad() ),
        BatchNorm(out_channels), 
        x -> relu.(x)
    )

end



function encoder_block( kernel, in_channels, out_channels )

    encoder    = encoder_module( in_channels )

    downsample = downsampler( kernel, in_channels, out_channels )

    return Chain( encoder, downsample )

end



function decoder_block( kernel, in_channels, out_channels )

    decoder  = decoder_module( in_channels )

    upsample = upsampler( kernel, in_channels, out_channels )

    return Chain( decoder, upsample )

end



function inception_coder( block_type, kernels, channels )

    return Chain( [ block_type(kernels[i], channels[i], channels[i+1] ) for i in 1:length(kernels) ]... )

end



function pre_encoder( out_channels )

    return Chain(

        downsampler( 3, 3, 4 ), 
        downsampler( 3, 4, 8 )

    )

end


function post_decoder( in_channels )

    return Chain(

        upsampler( 4, in_channels, 8 ),
        upsampler( 4, 8, 3 )
        # ConvTranspose( (3, 3), )
    )

end



function inception_encoder(; model_size=128, kernels=[4, 4], channels=[3, 32, model_size] )

    pre_encoder_ = pre_encoder( channels[1] )

    encoder      = inception_coder( encoder_block, kernels, channels )

    mean_pool    = GlobalMeanPool()

    reshaper     = x -> permutedims(x, (3, 1, 2, 4))

    return Chain( encoder..., mean_pool, reshaper ) 

end



function inception_decoder(; model_size=128, kernels=[4, 4], channels=[model_size, 32, 4] )

    reshaper      = x -> permutedims( x, (2, 3, 1, 4) )

    decoder       = inception_coder( decoder_block, kernels, channels )

    post_decoder_ = post_decoder( channels[end] )

    return Chain( reshaper, decoder..., post_decoder_... )

end



export inception_encoder, inception_decoder

end
