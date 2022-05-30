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

    init_model( "data/models/audio.bson", create_audio_autoencoder() )
    serialize( "data/audio/checkpoint", (0, 0, 0))

end

if args["init-video"]

    init_model( "data/models/video.bson", create_video_autoencoder() )
    serialize( "data/video/checkpoint", (0, 0, 0))

end

if args["train-audio"]

    train_autoencoder( "data/models/audio.bson", "data/audio", AudioIterator )

end

if args["train-video"]

    train_autoencoder( "data/models/video.bson", "data/video", VideoIterator )

end

if args["test-audio"]

    test_autoencoder("data/models/audio.bson", "data/audio", AudioIterator, save_audio, "audio_test.mp4" )

end

if args["test-video"]

    test_autoencoder("data/models/video.bson", "data/video", VideoIterator, save_video, "video_test.mp4" )

end

