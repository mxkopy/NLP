using Flux, Serialization, WAV, Zygote, Distributions, CUDA
# https://github.com/koshian2/inception-vae/blob/master/vae_model.py



function encoder_conv(in_channels, out_channels, kernel)

    return Chain(
        
        Conv( (kernel, kernel), in_channels => out_channels, pad=(kernel-1)÷2 ),
        BatchNorm( out_channels ),
        x -> relu.(x)
    )

end



function decoder_conv( in_channels, out_channels, kernel )

    return Chain(
        
        ConvTranspose( (kernel, kernel), in_channels => out_channels, pad=(kernel-1)÷2 ),
        BatchNorm( out_channels ),
        x -> relu.(x)
    )

end



function encoder_module( channels )

    bn_channels = channels ÷ 2

    bottleneck = encoder_conv( channels, bn_channels, 1)

    conv1 = encoder_conv( bn_channels, channels, 1 )
    conv3 = encoder_conv( bn_channels, channels, 3 )
    conv5 = encoder_conv( bn_channels, channels, 5 )
    conv7 = encoder_conv( bn_channels, channels, 7 )

    pool3 = MaxPool( (3, 3), stride=1, pad=1)
    pool5 = MaxPool( (5, 5), stride=1, pad=2)

    bn      = x -> bottleneck(x)

    pre     = x -> ( conv1( bn(x) ), conv3( bn(x) ), conv5( bn(x) ), conv7( bn(x) ), pool3( x ), pool5( x ) )

    forward = x -> cat( pre( x )..., dims=3 )

    return forward, (bottleneck..., conv1..., conv3..., conv5..., conv7..., pool3, pool5)

end



function decoder_module( channels )

    bn_channels = channels ÷ 4

    bottleneck = decoder_conv( channels, bn_channels, 1)

    conv1 = decoder_conv( bn_channels, channels, 1 )
    conv3 = decoder_conv( bn_channels, channels, 3 )
    conv5 = decoder_conv( bn_channels, channels, 5 )
    conv7 = decoder_conv( bn_channels, channels, 7 )

    pool3 = MaxPool( (3, 3), stride=1, pad=1)
    pool5 = MaxPool( (5, 5), stride=1, pad=2)

    bn      = x -> bottleneck(x)

    forward = x -> conv1( bn(x) ) + conv3( bn(x) ) + conv5( bn(x) ) + conv7( bn(x) ) + pool3( x ) + pool5( x )

    return forward, (bottleneck..., conv1..., conv3..., conv5..., conv7..., pool3, pool5)

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

    encoder, params = encoder_module( in_channels )

    downsample      = downsampler( in_channels * 6, out_channels, kernel )

    return x -> downsample( encoder(x) ), (params..., downsample...)

end



function decoder_block( in_channels, out_channels, kernel )

    decoder, params = decoder_module( in_channels )

    upsample        = upsampler( in_channels, out_channels, kernel )

    return x -> upsample( decoder( x ) ), (params..., upsample...)

end



function inception_coder( block_type, channels, kernels )

    blocks = map( 1:length(kernels) ) do i

        return block_type( channels[i], channels[i+1], kernels[i] )

    end

    layers = [ layer for (layer, _) in blocks ]
    params = [ param for (_, param) in blocks ]

    forward = foldl( layers ) do cumulative, layer

        return x -> layer( cumulative( x ) )

    end

    return forward, params

end



function inception_encoder(model_size; channels=[4, 8, 16, 32, 64, 128], kernels=[4, 4, 4, 4, 2] )

    encode, params  = inception_coder( encoder_block, channels, kernels )

    conv            = Conv((1, 1), 3=>channels[1], pad=SamePad())

    pool            = AdaptiveMeanPool( (1, 1) )

    encoder         = x -> x |> conv |> encode |> pool |> x -> reshape(x, (model_size, 1, 1, :)) |> softmax

    return encoder, (params..., conv, pool)

end



function inception_decoder(sample_size; channels=[128, 64, 32, 16, 8, 4], kernels=[4, 4, 4, 4, 4] )

    decode, params  = inception_coder( decoder_block, channels, kernels )

    conv            = ConvTranspose( (1, 1), channels[end] => 3, pad=SamePad() )

    pool            = AdaptiveMeanPool( (sample_size, sample_size ) )

    decoder         = x -> reshape(x, (1, 1, size(x)[1], :) ) |> decode |> conv |> pool 

    return decoder, (params..., conv, pool)

end
