include("AutoEncoder.jl")
include("Model.jl")
include("Data.jl")


# Runs a single training iteration with backprop. 

function train_iter( model, parameters, opt, data, model_size=128 )

    # Generates an array of random floats. These are used for the reparameterization trick

    unit_gaussians = rand( Normal( 1.0, 0.1 ), model_size )

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

reshape_audio =  x -> reshape(x, (size(x)[1], 1, size(x)[2], :) )
reshape_video =  x -> reshape(x, (size(x)[3], size(x)[2], size(x)[1], :) )



function train_over_data( model, parameters, opt, iterator, reshape_data )

    for x in iterator

        data = reshape_data( x )

        train_iter( model, parameters, opt, data )

    end

end



function train_audio()

    model, parameters, opt = load_model("data/models/audio.bson")

    num_files = length(readdir("data/audio"))

    for (i, dir) in enumerate(readdir("data/audio", join=true))

        println("$i of $num_files")

        train_over_data( model, parameters, opt, AudioIterator(dir), reshape_audio )

        save_model("data/models/audio.bson", model, parameters, opt)

    end

end


function train_video()

    model, parameters, opt = load_model("data/models/video.bson")

    num_files = length(readdir("data/video"))

    for (i, dir) in enumerate(readdir("data/video", join=true))

        println("$i of $num_files")

        train_over_data( model, parameters, opt, VideoIterator(dir), reshape_video )

        save_model("data/models/video.bson", model, parameters, opt)

    end

end