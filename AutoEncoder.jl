include("Model.jl")
include("InceptionVAE.jl")
include("AudioVAE.jl")

using Flux, Serialization, WAV, Zygote, Distributions, CUDA, .InceptionVAE, .AudioVAE
 


mutable struct AutoEncoder

    encoder::Chain
    decoder::Chain
    mean::Dense
    std::Dense

end



function (model::AutoEncoder)(param, data)

    enc_out    = model.encoder( data )

    means      = model.mean( enc_out )
    devs       = model.std( enc_out )

    latent     = ( param .* devs ) + means

    dec_out    = model.decoder( latent )

    return enc_out, means, devs, latent, dec_out

end



function to_device(model::AutoEncoder, device)

    model.encoder = model.encoder |> device
    model.decoder = model.decoder |> device
    model.mean    = model.mean    |> device
    model.std     = model.std     |> device

end



Flux.@functor AutoEncoder



function create_audio_autoencoder( model_size=128, audio_size=1764 )

    encoder = audio_encoder( model_size, audio_size )
    decoder = audio_decoder( model_size, audio_size )
    
    mean    = Dense( model_size, model_size )
    std     = Dense( model_size, model_size )

    return AutoEncoder( encoder, decoder, mean, std ) 

end    



function create_video_autoencoder( model_size=128, image_size=640 )

    encoder  = inception_encoder( model_size )
    decoder  = inception_decoder( image_size )

    mean     = Dense( model_size, model_size )
    std      = Dense( model_size, model_size )

    return AutoEncoder( encoder, decoder, mean, std ) 

end

