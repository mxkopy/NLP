include("Model.jl")
include("InceptionVAE.jl")
include("AudioVAE.jl")

using Flux, Serialization, WAV, Zygote, Distributions, CUDA, .InceptionVAE, .AudioVAE
 


mutable struct AutoEncoder

    encoder::Chain
    decoder::Chain
    mean::Dense
    std::Dense
    dropout::Dropout

end



function (model::AutoEncoder)(param, data)

    enc_out    = model.encoder( data ) |> model.dropout

    enc_out    = permutedims( enc_out, (3, 1, 2, 4)) |> softmax

    means      = model.mean( enc_out )
    devs       = model.std( enc_out )

    latent     = ( param .* devs ) + means

    latent     = permutedims( latent, (2, 3, 1, 4) )

    dec_out    = model.decoder( latent )

    return dec_out

end



function to_device(model::AutoEncoder, device)

    model.encoder = model.encoder |> device
    model.decoder = model.decoder |> device
    model.mean    = model.mean    |> device
    model.std     = model.std     |> device
    model.dropout = model.dropout |> device

end



Flux.@functor AutoEncoder



function create_audio_autoencoder( model_size=128, audio_size=1764 )

    encoder = audio_encoder( model_size, audio_size )
    decoder = audio_decoder( model_size, audio_size )
    
    mean    = Dense( model_size, model_size, celu )
    std     = Dense( model_size, model_size, celu )

    return AutoEncoder( encoder, decoder, mean, std, Dropout(0.5) ) 

end    



function create_video_autoencoder( model_size=128, image_size=640 )

    encoder  = inception_encoder( model_size )
    decoder  = inception_decoder( image_size )

    mean     = Dense( model_size, model_size, celu )
    std      = Dense( model_size, model_size, celu )

    return AutoEncoder( encoder, decoder, mean, std, Dropout(0.5) ) 

end



function init_audio_autoencoder( savename, model_size, audio_size )

    opt = ADAM( 0.01 )

    model = create_audio_autoencoder( model_size, audio_size )

    serialize( savename, (model, opt, Flux.params(model.encoder, model.decoder, model.mean, model.std, model.dropout), args) )

end



function init_video_autoencoder( savename, args )

    opt = ADAM( 0.01 )

    model = create_video_autoencoder( args["model-size"], args["image-size"] )

    serialize( savename, (model, opt, Flux.params(model.encoder, model.decoder, model.mean, model.std, model.dropout), args) )

end
