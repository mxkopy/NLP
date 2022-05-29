include("AutoEncoder.jl")
include("Model.jl")
include("Data.jl")

# Runs a single training iteration with backprop. 

function train_iter( model, parameters, opt, data, model_size=128 )

    # Generates an array of random floats. These are used for the reparameterization trick

    unit_gaussians = rand( Normal( 1.0, 0.1 ), model_size ) |> gpu

    r_loss, d_loss = 0, 0

    gs = gradient( parameters ) do

        r_loss, d_loss = loss_function( model..., unit_gaussians, data )

        return 2 * r_loss + d_loss

    end

    println('\n', "r ", string(r_loss), " | d ", string(d_loss) )

    Flux.Optimise.update!( opt, parameters, gs )

end



function save_model( savename, model, parameters, opt )

    serialize( savename, ( model, parameters, opt ) )

end

# This function deserializes the model and relevant parameters from the disk, 
    # and sets the data IOPipe to the checkpointed position. 

# Loading is fairly cheap, since it's done infrequently.

function load_model( savename )

    return deserialize( savename )

end


function init_model( model, savename )

    opt = ADAM( 0.01 )
    parameters = Flux.params( model... )

    save_model( savename, model, parameters, opt )

end

init_audio_model = init_model(create_audio_autoencoder(), "data/models/audio.bson")
init_video_model = init_model(create_video_autoencoder(), "data/models/video.bson")



function train_over_data( model, parameters, opt, iterator, reshape_data )

    for x in iterator

        data = reshape_data( x ) .|> Float32 |> gpu

        train_iter( model, parameters, opt, data )

    end

end



function train_autoencoder( model_dir, data_dir, data_itr, reshaper )

    model, parameters, opt = load_model( model_dir )

    model                  = model .|> gpu

    num_files = length(readdir(data_dir))

    for (i, dir) in enumerate(readdir(data_dir, join=true))

        println("$i of $num_files")

        train_over_data( model, parameters, opt, data_itr(dir), reshaper )

        model = model .|> cpu

        save_model(model_dir, model, parameters, opt)

    end

end



function test_audio(; model_dir="data/models/audio.bson", output="audio_test.wav", data_dir="data/audio", model_size=128 )

    model, parameters, opt = load_model( model_dir )

    encoder, decoder, mean, std = model

    directory   = readdir( data_dir, join=true )[1]

    audio_itr   = AudioIterator( directory )

    _, fs, _, _ = wavread( directory )

    out = map( audio_itr ) do sample
    
        return eval_model( encoder, decoder, mean, std, ones(model_size), reshape_audio(sample) )

    end

    out = cat(out..., dims=1)

    out = reshape( out, ( size(out)[1], : ) )

    wavwrite( out, output, Fs=fs )

end



function test_video(; model_dir="data/models/video.bson", output="video_test.mp4", data_dir="data/video", model_size=128 )

    model, parameters, opt = load_model( model_dir )

    encoder, decoder, mean, std = model

    directory   = readdir( data_dir, join=true )[1]

    video_itr   = VideoIterator( directory )

    out = map( video_itr ) do data

        img_out = eval_model( encoder, decoder, mean, std, ones(model_size), data )

        img_out = mapslices( img_out, dims=4 ) do img

            return reshape( img, (size(img)[3], size(img)[2], size(img)[1]) ) |> colorview

        end

    end

    VideoIO.save(output, out)

end