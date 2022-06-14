module AudioVAE

using Flux, CUDA


function dilated_conv_block( dilation; kernel=256, in_channels=2, out_channels=2 )

    conv_f     = Conv((kernel, 1), in_channels => in_channels, dilation=dilation, pad=SamePad())
    conv_g     = Conv((kernel, 1), in_channels => in_channels, dilation=dilation, pad=SamePad())

    conv_F     = Chain( conv_f, x-> tanh.(x) )
    conv_G     = Chain( conv_g, x-> sigmoid.(x) )

    conv       = Conv( (1, 1), in_channels => out_channels )

    gated_conv = Parallel( (l, r) -> l .* r, conv_F, conv_G )

    return Chain( SkipConnection( gated_conv, + ), conv ) 

end


function downsampler( in_channels )

    return Chain( 

        Conv((3, 1), in_channels => 4, stride=2, pad=SamePad()), 
        Conv((3, 1), 4 => 2,           stride=2, pad=SamePad()), 
        Conv((3, 1), 2 => 1,           stride=2, pad=SamePad())

    )

end


function upsampler( out_channels )

    return Chain( 

        ConvTranspose((3, 1), 1 => 2,            stride=2, pad=SamePad()),
        ConvTranspose((3, 1), 2 => 4,            stride=2, pad=SamePad()),
        ConvTranspose((3, 1), 4 => 5,            stride=2, pad=SamePad()), 
        ConvTranspose((3, 1), 5 => out_channels, stride=2, pad=SamePad())

    )

end


function audio_encoder( model_size, out_channels=6 )

    convolutions = Parallel( +, 

        dilated_conv_block( 1,  out_channels=out_channels ),
        dilated_conv_block( 7,  out_channels=out_channels ), 
        dilated_conv_block( 17, out_channels=out_channels ),
        dilated_conv_block( 63, out_channels=out_channels ) 

    )

    pool = AdaptiveMeanPool((model_size, 1))

    return Chain( convolutions, downsampler(out_channels), pool )

end


function audio_decoder( audio_size, in_channels=6 )

    convolutions = Parallel( +, 

        dilated_conv_block( 1,   in_channels=in_channels ),
        dilated_conv_block( 7,   in_channels=in_channels ), 
        dilated_conv_block( 17,  in_channels=in_channels ),
        dilated_conv_block( 63,  in_channels=in_channels )

    )

    pool = AdaptiveMeanPool((audio_size, 1))

    return Chain( upsampler(in_channels), convolutions, pool )

end


export audio_encoder, audio_decoder

end










