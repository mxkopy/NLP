module InceptionVAE


using Flux, Serialization, WAV, Zygote, Distributions, CUDA


# TODO: encoder_conv and decoder_conv are really similar, and could also be useful 
# in the audio autoecoder. maybe it's a good idea to consolidate them somehow?
function encoder_conv( in_channels, out_channels, kernel )

    return Chain(
        
        Conv( kernel, in_channels => out_channels, pad=SamePad() ),
        BatchNorm( out_channels ),
        x -> relu.(x)
    )

end



function decoder_conv( in_channels, out_channels, kernel )

    return Chain(
        
        ConvTranspose( kernel, in_channels => out_channels, pad=SamePad() ),
        BatchNorm( out_channels ),
        x -> relu.(x)
    )

end



function conv_group( conv_type, channels, bottleneck_channels, connection )

    bn      = conv_type( channels, bottleneck_channels, (1, 1) )

    conv1   = conv_type( bottleneck_channels, channels, (1, 1) )
    conv3   = conv_type( bottleneck_channels, channels, (3, 3) )

    conv5   = conv_type( bottleneck_channels, bottleneck_channels, (3, 3) )
    conv5v  = conv_type( bottleneck_channels, channels, (3, 1) )
    conv5h  = conv_type( bottleneck_channels, channels, (1, 3) )

    pool3   = MaxPool( (3, 3), stride=1, pad=1)
    conv1_p = conv_type( bottleneck_channels, channels, (1, 1) )

    g1    = Parallel( connection, conv5h, conv5v )
    c1    = Chain( conv5, g1 )

    g2    = Parallel( connection, conv1, conv3, c1, conv1_p ∘ pool3 )
    c2    = Chain( bn, g2 )

    return c2

end


function encoder_module( channels )

    # connection in the paper is (x...) -> cat( x..., dims=3), but that is super resource intensive
    return conv_group( encoder_conv, channels, channels ÷ 2, + )

end



function decoder_module( channels )

    return conv_group( decoder_conv, channels, channels ÷ 4, + )

end



function downsampler( in_channels, out_channels, kernel )

    return Chain( 

        MeanPool( (kernel, kernel) ), 

        Conv( (1, 1), in_channels => out_channels ),

        BatchNorm( out_channels ), 
        
        x -> relu.(x)
    )

end



function upsampler( in_channels, out_channels, kernel )

    return Chain( 

        ConvTranspose( (kernel, kernel), in_channels => out_channels, stride=kernel ),

        BatchNorm( out_channels ), 
        
        x -> relu.(x)
    )

end



function encoder_block( in_channels, out_channels, kernel )

    encoder    = encoder_module( in_channels )

    downsample = downsampler( in_channels, out_channels, kernel )

    return Chain( encoder, downsample )

end



function decoder_block( in_channels, out_channels, kernel )

    decoder  = decoder_module( in_channels )

    upsample = upsampler( in_channels, out_channels, kernel )

    return Chain( decoder, upsample )

end



function inception_coder( block_type, channels, kernels )

    return Chain( [ block_type(channels[i], channels[i+1], kernels[i] ) for i in 1:length(kernels) ]... )

end



function pre_encoder( out_channels )

    return Chain(

        Conv((3, 3), 3 => 4,            stride=2),
        Conv((3, 3), 4 => 8,            stride=2),
        Conv((3, 3), 8 => out_channels, stride=2)

    )

end


function post_decoder( in_channels, image_size )

    return Chain(

        ConvTranspose((4, 4), in_channels=>8, stride=4),
        ConvTranspose((4, 4), 8=>4,           stride=4),
        ConvTranspose((4, 4), 4=>3,           stride=4),
        AdaptiveMeanPool((image_size, image_size))
    )

end



function inception_encoder(model_size; channels=[32, model_size], kernels=[4] )

    encode = inception_coder( encoder_block, channels, kernels )

    return Chain( pre_encoder( channels[1] ), encode..., GlobalMeanPool() ) 

end



function inception_decoder(image_size; model_size=128, channels=[model_size, 64, 32], kernels=[4, 4] )

    decode = inception_coder( decoder_block, channels, kernels )

    return Chain( decode..., post_decoder( channels[end], image_size ) )

end



export inception_encoder, inception_decoder

end