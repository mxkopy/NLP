include("AutoEncoder.jl")
include("Model.jl")
include("Data.jl")

using Printf


function loss_function( model, param, data, resizer ) 

    enc_out, means, devs, latent, dec_out = model(param, data)

    MSE = Flux.Losses.mse( resizer(dec_out), data )

    KLD = Flux.Losses.kldivergence( means, devs )
    
    return MSE, KLD

end



function backprop_iteration( model, opt, parameters, param, data, resizer )

    r_loss, d_loss = 0, 0

    gs = gradient( parameters ) do

        r_loss, d_loss = loss_function( model, param, data, resizer )

        return r_loss + 0.5 * d_loss

    end

    Flux.Optimise.update!( opt, parameters, gs )

    return r_loss, d_loss

end


mutable struct Trainer

    model::AutoEncoder
    optimizer
    parameters
    checkpoint

end

function save( savename, trainer::Trainer )

    to_device( trainer.model, cpu )

    serialize( savename, trainer )

end



function init_trainer( model_creator, model_size, data_size, filename, optimizer, device )

    model   = model_creator( model_size, data_size )
    trainer = Trainer( model, optimizer, Flux.params( model.encoder, model.decoder, model.mean, model.std), 0 )

    serialize( filename, trainer )

end



function AudioTrainer(; model_size=128, audio_size=1764, filename="data/models/audio", optimizer=ADAM(0.01), device=gpu )
    
    init_trainer( create_audio_autoencoder, model_size, audio_size, filename, optimizer, device )

end

function VideoTrainer(; model_size=128, image_size=640, filename="data/models/video.bson", optimizer=ADAM(0.01), device=gpu )
    
    init_trainer( create_video_autoencoder, model_size, image_size, filename, optimizer, device )

end


function train_loop( trainer::Trainer, iterator, savename; save_freq=1000 )

    resizer = AdaptiveMeanPool( iterator.data_size )

    for (i, (param, data)) in enumerate( Iterators.drop(iterator, trainer.checkpoint) )

        r_loss, d_loss = backprop_iteration( trainer.model, trainer.optimizer, trainer.parameters, param, data, resizer )

        next_save = save_freq - ( i % save_freq ) - 1 

        @printf "\rr %.5e d %.5e next_save %d     " r_loss d_loss next_save
        flush(stdout)

        if next_save == 0

            trainer.checkpoint += save_freq

            save(savename, trainer)

            to_device(trainer.model, iterator.device)

        end

    end

end



function save_audio( data, output )

    file = reduce( (l, r) -> cat( l, r, dims=1 ), data )

    file = reshape(file, (size(file)[1], :))

    wavwrite(file, output, Fs=44100)

end



function save_video( data, output )

    file = map( data ) do img

        img = reshape( img, (size(img)[3], size(img)[2], size(img)[1]) )
        img = N0f8.( image )
        img = colorview(RGB, image)

        return img

    end

    VideoIO.save(output, file)

end



function test_autoencoder( model, data_iterator, save_function, output, num_iter )

    testmode!( model )

    to_device( model, data_iterator.device )

    

    data = mapreduce( (l, r) -> cat( l, r, dims=1 ), Iterators.take( data_iterator, num_iter )) do (param, data)

        _, _, _, _, y = model( param, data )

        return y

    end

    save_function(data, output)

end

