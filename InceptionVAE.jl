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


function conv_group( conv_type, channels, bottleneck_channels, connection )

    bn    = conv_type( channels, bottleneck_channels, 1 )

    conv1 = conv_type( bottleneck_channels, channels, 1 )
    conv3 = conv_type( bottleneck_channels, channels, 3 )
    conv5 = conv_type( bottleneck_channels, channels, 5 )
    conv7 = conv_type( bottleneck_channels, channels, 7 )

    pool3 = MaxPool( (3, 3), stride=1, pad=1)
    pool5 = MaxPool( (5, 5), stride=1, pad=2) 


    g1    = Parallel( connection, conv1, conv3, conv5, conv7 )

    c1    = Chain( bn, g1 )

    g2    = Parallel( connection, c1, pool3, pool5 ) 

    return g2

end


function encoder_module( channels )

    return conv_group( encoder_conv, channels, channels ÷ 2, (x...) -> cat( x..., dims=3) )

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

    downsample = downsampler( in_channels * 6, out_channels, kernel )

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



function inception_encoder(model_size; channels=[4, 8, 16, 32, 64, 128], kernels=[4, 4, 4, 4, 2] )

    encode = inception_coder( encoder_block, channels, kernels )

    conv   = Conv((1, 1), 3=>channels[1], pad=SamePad())

    pool   = AdaptiveMeanPool( (1, 1) )

    return Chain( conv, encode, pool, x -> reshape(x, (model_size, 1, 1, :)), softmax ) 

end



function inception_decoder(sample_size; channels=[128, 64, 32, 16, 8, 4], kernels=[4, 4, 4, 4, 4] )

    decode = inception_coder( decoder_block, channels, kernels )

    conv   = ConvTranspose( (1, 1), channels[end] => 3, pad=SamePad() )

    pool   = AdaptiveMeanPool( (sample_size, sample_size ) )

    return Chain( x -> reshape( x, (1, 1, size(x)[1], :) ), decode, conv, pool )

end
