using ArgParse

include("Training.jl")

function arguments()

    s = ArgParseSettings()

    @add_arg_table s begin 

        "--init"
            help   = "reinitializes & saves autoencoders"
            action = :store_true

        "--train-autoencoders"
            help   = "trains autoencoders on the data/audio & data/video directories"
            action = :store_true

    end

    return parse_args( s )

end

args = arguments()

if args["init"]

    init_model( create_audio_autoencoder(), "data/models/audio.bson" )
    println("audio model initialized")

    init_model( create_video_autoencoder(), "data/models/video.bson" )
    println("video model initialized")


end

if args["train-autoencoders"]

    train_audio()
    println("audio model training")

    train_video()
    println("video model training")


end