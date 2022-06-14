include("InceptionVAE.jl")
include("AudioVAE.jl")

using Flux, Serialization, WAV, Zygote, Distributions, CUDA, .InceptionVAE, .AudioVAE
 


mutable struct AutoEncoder

    encoder
    decoder
    mean
    std

end



function (model::AutoEncoder)(param, data)

    enc_out    = model.encoder( data )

    means      = model.mean( enc_out )
    devs       = model.std( enc_out )

    latent     = ( param .* devs ) + means

    dec_out    = model.decoder( latent )

    return enc_out, means, devs, latent, dec_out

end


Flux.@functor AutoEncoder
# Flux.trainable( c::AutoEncoder ) = (c.encoder, c.mean, c.std, c.decoder)



function create_audio_autoencoder( model_size=128, audio_size=1764 )

    encoder = audio_encoder( model_size )
    decoder = audio_decoder( audio_size )
    
    mean    = Dense( model_size, model_size )
    std     = Dense( model_size, model_size )

    return AutoEncoder( encoder, decoder, mean, std ) 

end    



function create_video_autoencoder( model_size=128 )

    encoder  = inception_encoder( model_size=model_size )
    decoder  = inception_decoder( model_size=model_size )

    mean     = Dense( model_size, model_size )
    std      = Dense( model_size, model_size )

    return AutoEncoder( encoder, decoder, mean, std ) 

end

