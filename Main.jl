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

        "--test-iterations"
            arg_type = Int
            default  = 1000

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

    iterator = AudioIterator( program_args["audio-data"],program_args["audio-size"], batches=program_args["batches"] )

    to_device(trainer.model, iterator.device)

    train_loop(trainer, iterator, program_args["video-model-filename"], save_freq=program_args["save-freq"])

end

if program_args["train-video"]

    trainer  = deserialize( program_args["video-model-filename"] )

    iterator = VideoIterator( program_args["video-data"], program_args["image-size"], batches=program_args["batches"] )

    to_device(trainer.model, iterator.device)

    train_loop(trainer, iterator, program_args["video-model-filename"], save_freq=program_args["save-freq"])

end

if program_args["test-audio"]

    trainer = deserialize(program_args["audio-model-filename"])
    model   = trainer.model
    itr     = AudioIterator( program_args["audio-data"], program_args["audio-size"], batches=1, rand_dist=[0], model_size=program_args["model-size"] ) 
    test_autoencoder(model, itr, save_audio, "audio_test.wav", program_args["test-iterations"] )

end

if program_args["test-video"]

    trainer = deserialize(program_args["video-model-filename"])
    model   = trainer.model
    itr     = VideoIterator( program_args["video-data"], program_args["video-size"], batches=1, rand_dist=[0], model_size=program_args["model-size"] ) 
    test_autoencoder(model, itr, save_video, "video_test.mp4", program_args["test-iterations"] )

end
