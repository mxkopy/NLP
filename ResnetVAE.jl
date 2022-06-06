using Flux, Serialization, Zygote, Distributions, CUDA


function conv_block( conv_type, kernel, in_channels, out_channels, stride=1 )

    return Chain(
        
        conv_type( kernel, in_channels => out_channels, pad=SamePad(), stride=stride ),
        BatchNorm( out_channels ),
        x -> relu.(x)
    )

end



function residual_block( layer, connection=(mx, x) -> cat(mx, x, dims=3) )

    return SkipConnection( layer, connection )

end



function encoder_block( in_channels, out_channels, output_size )

    return Chain( 
        
        residual_block( conv_block( Conv, (3, 3), in_channels, out_channels ) ),
        Conv( (1, 1), in_channels + out_channels=>out_channels),
        AdaptiveMeanPool( (output_size, output_size) )

    )
    
end


function decoder_block( in_channels, out_channels, upsample )

    return Chain( 
        
        residual_block( conv_block( ConvTranspose, (3, 3), in_channels, out_channels ) ),
        Conv( (1, 1), in_channels + out_channels=>out_channels), 
        Upsample(upsample)

    )

end


function resnet_encoder(; model_size=128 )

    return Chain(

        encoder_block(3, 16, 64),
        encoder_block(16, model_size, 4), 
        GlobalMeanPool(), 
        x -> permutedims(x, (3, 1, 2, 4))

    )

end


function resnet_decoder(; model_size=128 )

    return Chain(

        x -> permutedims( x, (2, 3, 1, 4) ), 
        Upsample( 16 ),
        decoder_block(model_size, 16, 4), 
        decoder_block(16, 3, 4)

    )

end