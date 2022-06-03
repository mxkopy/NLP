include("AutoEncoder.jl")
include("Model.jl")
include("Data.jl")

using Printf


function loss_function( model, param, x ) 

    y = model(param, x)
    
    return Flux.Losses.mse( y, x ), Flux.Losses.kldivergence( softmax( y ), softmax( x ) )

end



function backprop_iteration( model, opt, parameters, param, data )

    r_loss, d_loss = 0, 0

    gs = gradient( parameters ) do

        r_loss, d_loss = loss_function( model, param, data )

        return 2 * r_loss + d_loss

    end

    Flux.Optimise.update!( opt, parameters, gs )

    return r_loss, d_loss

end


mutable struct Trainer

    model::AutoEncoder
    optimizer
    parameters
    model_size
    device
    checkpoint

end

function save( savename, trainer::Trainer )

    to_device( trainer.model, cpu )

    serialize( savename, trainer )

end


function train_iteration( trainer::Trainer, data )

    data           = data .|> Float32 |> trainer.device

    param          = rand( Normal( 1.0, 0.1 ), trainer.model_size ) .|> Float32 |> trainer.device

    r_loss, d_loss = backprop_iteration( trainer.model, trainer.optimizer, trainer.parameters, param, data )

    return r_loss, d_loss

end


function init_trainer( model_creator, model_size, data_size, filename, optimizer, device )

    model   = model_creator( model_size, data_size )
    trainer = Trainer( model, optimizer, Flux.params( model.encoder, model.decoder, model.mean, model.std), model_size, device, 0 )

    serialize( filename, trainer )

end



function AudioTrainer(; model_size=128, audio_size=1764, filename="data/models/audio", optimizer=ADAM(0.01), device=gpu )
    
    init_trainer( create_audio_autoencoder, model_size, audio_size, filename, optimizer, device )

end

function VideoTrainer(; model_size=128, image_size=640, filename="data/models/video.bson", optimizer=ADAM(0.01), device=gpu )
    
    init_trainer( create_audio_autoencoder, model_size, image_size, filename, optimizer, device )

end


function train_loop( trainer::Trainer, iterator, savename; save_freq=1000 )

    for (i, data) in enumerate( iterator )

        r_loss, d_loss = train_iteration( trainer, data )

        next_save = save_freq - ( i % save_freq ) - 1 

        @printf "\rr %.5e d %.5e next_save %d" r_loss d_loss next_save
        flush(stdout)

        if next_save == 0

            trainer.checkpoint += save_freq

            save(savename, trainer)

            to_device(trainer.model, trainer.device)

        end

    end

end


# function train_autoencoder( model_dir, data_dir, iterator_type, args; device=gpu )

#     model, opt, parameters, args    = deserialize( model_dir )

#     n, r_avg, d_avg                 = deserialize( data_dir * "/checkpoint" )

#     data_itr                        = Iterators.drop( iterator_type( data_dir, iterator_type, args ), n )

#     to_device( model, gpu )

#     for (i, data) in enumerate( data_itr )

#         param = rand( Normal( 1.0, 0.1 ), args["model-size"] )

#         param = param .|> Float32 |> device
#         data  = data  .|> Float32 |> device

#         r_loss, d_loss = train_iter( model, opt, parameters, param, data )

#         r_avg, d_avg   = (r_avg + r_loss) / (n + i), (d_avg + d_loss) / (n + i)

#         next_save      = args["save-freq"] - ( (n + i) % args["save-freq"] ) - 1 

#         @printf "\rr %.5e d %.5e r_avg %.5e d_avg %.5e next_save %d" r_loss d_loss r_avg d_avg next_save
#         flush(stdout)

#         if next_save == 0
        
#             println("model saved!")

#             to_device( model, cpu )
            
#             serialize( model_dir, (model, opt, parameters, args) )
#             serialize( data_dir * "/checkpoint", (n + i, r_avg, d_avg ) )

#             to_device( model, gpu )

#         end

#     end

# end



function save_audio( data, output )

    file = cat( data..., dims=1 )

    file = reshape(file, (size(file)[1], :))

    wavwrite(file, output, Fs=44100)

end



function save_video( data, output )

    file = map( data ) do img

        return reshape( img, (size(img)[3], size(img)[2], size(img)[1]) ) |> colorview

    end

    VideoIO.save(output, file)

end



function test_autoencoder( model, data_dir, data_iterator, save_function, output, data_size, num_iter=10000 )

    model.dropout = x -> x

    data_itr      = Iterators.take( directory_iterator( data_dir, data_iterator, data_size ), num_iter )

    out = map( data_itr ) do data 

        return eval_model( (encoder, decoder, mean, std), ones(data_size), data )

    end

    save_function(out, output)

end

