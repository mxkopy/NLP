using ArgParse

include("Training.jl")

function arguments()

    s = ArgParseSettings()

    @add_arg_table s begin 

        "--init-audio"
            action = :store_true

	    "--init-video"
	        action = :store_true

        "--train-audio"
            action = :store_true
	
	    "--train-video"
	        action = :store_true

        "--test-video"
	        action = :store_true

        "--test-audio"
	        action = :store_true


    end

    return parse_args( s )

end

args = arguments()

if args["init-audio"]

    init_model( create_audio_autoencoder(), "data/models/audio.bson" )

end

if args["init-video"]

    init_model( create_video_autoencoder(), "data/models/video.bson" )

end

if args["train-audio"]

    train_autoencoder( "data/models/audio.bson", "data/audio", AudioIterator, reshape_audio )

end


if args["train-video"]

    train_autoencoder( "data/models/video.bson", "data/video", VideoIterator, reshape_video )

end


if args["test-video"]

    test_video()

end

if args["test-audio"]

    test_audio()

end