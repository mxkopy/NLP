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
    iterator::BatchIterator
    device

end

function save( savename, trainer::Trainer )

    to_device( trainer.model, cpu )

    serialize( savename, trainer )

end


function train_iteration( trainer::Trainer )

    data           = Iterators.take( trainer.iterator, 1 ) .|> Float32 |> trainer.device

    param          = rand( Normal( 1.0, 0.1 ), trainer.args["model-size"] ) .|> Float32 |> trainer.device

    r_loss, d_loss = backprop_iteration( trainer.model, trainer.optimizer, trainer.parameters, param, data )

    return r_loss, d_loss

end

function Base.iterate( trainer::Trainer, state )

    isempty(trainer.iterator.itr) ? nothing : (train_iteration( trainer ), trainer.iterator.itr )

end

function Base.iterate( trainer::Trainer )

    to_device( trainer.model, trainer.device )

    return (0, 0), trainer.iterator.itr

end


function AudioTrainer(; model_size=128, audio_size=1764, data_dir="data/audio", batches=4, optimizer=ADAM(0.01), device=gpu )
    
    model   = create_audio_autoencoder( model_size, audio_size )
    itr     = AudioIterator( data_dir, audio_size, batches=batches )

    trainer = Trainer( model, optimizer, Flux.params( model.encoder, model.decoder, model.mean, model.std), itr, device )

    serialize( data_dir * "/audio.bson", trainer )

end

function VideoTrainer(; model_size=128, image_size=640, data_dir="data/video", batches=4, optimizer=ADAM(0.01), device=gpu )
    
    model   = create_video_autoencoder( model_size, image_size )
    itr     = VideoIterator( data_dir, image_size, batches=batches )

    trainer = Trainer( model, optimizer, Flux.params( model.encoder, model.decoder, model.mean, model.std), itr, device )

    serialize( data_dir * "/video.bson", trainer )

end


function train_loop( trainer::Trainer, savename; save_freq=1000 )

    for (i, (r_loss, d_loss)) in enumerate( trainer )

        next_save = save_freq - ( i % save_freq ) - 1 

        @printf "\rr %.5e d %.5e next_save %d" r_loss d_loss next_save
        flush(stdout)

        if next_save == 0

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

