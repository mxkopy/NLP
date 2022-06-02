include("Training.jl")

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

        "--batches"
            arg_type = Int
            default  = 1

        "--save-freq"
            arg_type = Int
            default  = 5000

        "--model-size"
            arg_type = Int
            default  = 128

        "--audio-size"
            arg_type = Int
            default  = 1764

        "--image-size"
            arg_type = Int
            default  = 640

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

    AudioTrainer( model_size=program_args["model-size"], audio_size=program_args["audio-size"], filename=program_args["audio-model-filename"] )

end

if program_args["init-video"]

    VideoTrainer( model_size=program_args["model-size"], image_size=program_args["image-size"], filename=program_args["video-model-filename"] )

end

if program_args["train-audio"]

    trainer  = deserialize( program_args["audio-model-filename"] )

    iterator = Iterators.drop( AudioIterator( program_args["audio-data"],program_args["audio-size"], batches=program_args["batches"] ), trainer.checkpoint )

    to_device(trainer.model, trainer.device)

    train_loop(trainer, iterator, filename, save_freq=program_args["save-freq"])

end

if program_args["train-video"]

    trainer  = deserialize( program_args["video-model-filename"] )

    iterator = Iterators.drop( VideoIterator( program_args["video-data"], program_args["image-size"], batches=program_args["batches"] ), trainer.checkpoint )

    to_device(trainer.model, trainer.device)

    train_loop(trainer, iterator, program_args["video-model-filename"], save_freq=program_args["save-freq"])

end

if program_args["test-audio"]

    model, _, _, loaded_args = deserialize("data/models/audio.bson")
    test_autoencoder(model, "data/audio", AudioIterator, save_audio, "audio_test.wav", loaded_args["audio-size"] )

end

if program_args["test-video"]

    model, _, _, loaded_args = deserialize("data/models/video.bson")
    test_autoencoder("data/models/video.bson", "data/video", VideoIterator, save_video, "video_test.mp4", loaded_args["video-size"] )

end
