include("AutoEncoderTraining.jl")

using ArgParse

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

        "--test-iterations"
            arg_type = Int
            default  = 1000

        "--batches"
            arg_type = Int
            default  = 1

        "--epochs"
            arg_type = Int
            default  = 1

        "--save-freq"
            arg_type = Int
            default  = 10

        "--model-size"
            arg_type = Int
            default  = 128

        "--audio-size"
            arg_type = Int
            default  = 1764

        "--image-size"
            arg_type = Int
            default  = 256

        "--audio-data"
            arg_type = String
            default  = "data/audio"

        "--video-data"
            arg_type = String
            default  = "data/video"

        "--audio-model-filename"
            arg_type = String
            default  = "data/models/audio.bson"

        "--video-model-filename"
            arg_type = String
            default  = "data/models/video.bson"


    end

    return parse_args( s )

end

program_args = arguments()


if program_args["init-audio"]

    model      = create_audio_autoencoder( program_args["model-size"], program_args["audio-size"] )
    optimizer  = ADAM(0.001)

    @save program_args["audio-model-filename"] model optimizer

end

if program_args["init-video"]

    model      = create_video_autoencoder( program_args["model-size"] )
    optimizer  = ADAM(0.01)

    @save program_args["video-model-filename"] model optimizer

end

if program_args["train-audio"]

    @load program_args["audio-model-filename"] model optimizer

    for epoch in 1:program_args["epochs"]

        itr = AudioIterator( program_args["audio-data"], program_args["audio-size"], batches=program_args["batches"], model_size=program_args["model-size"] )
        train_autoencoder(model, optimizer, itr, program_args["audio-model-filename"], save_freq=program_args["save-freq"], epochs=program_args["epochs"] )

    end

end

if program_args["train-video"]

    @load program_args["video-model-filename"] model optimizer

    for epoch in 1:program_args["epochs"]

        itr = VideoIterator( program_args["video-data"], program_args["image-size"], batches=program_args["batches"], model_size=program_args["model-size"] )
        train_autoencoder(model, optimizer, itr, program_args["video-model-filename"], save_freq=program_args["save-freq"], epochs=program_args["epochs"] )  

    end
end

if program_args["test-audio"]

    @load program_args["audio-model-filename"] model optimizer
    itr = AudioIterator( program_args["audio-data"], program_args["audio-size"], batches=1, rand_dist=[0], model_size=program_args["model-size"] ) 
    test_autoencoder(model, itr, save_audio, "audio_test.wav", program_args["test-iterations"] )

end

if program_args["test-video"]

    @load program_args["video-model-filename"] model optimizer
    itr = VideoIterator( program_args["video-data"], program_args["image-size"], batches=1, rand_dist=[0], model_size=program_args["model-size"] ) 
    test_autoencoder(model, itr, save_video, "video_test.mp4", program_args["test-iterations"] )

end
