include("AutoEncoder.jl")
include("Model.jl")
include("Data.jl")

function train_audio()

    model, parameters, opt = load("data/models/audio.bson")

    num_files = length(readdir("data/audio"))

    for (i, dir) in enumerate(readdir("data/audio", join=true))

        println("$i of $num_files")

        train_over_data( model, parameters, opt, AudioIterator(dir), reshape_audio )

    end

    save("data/models/audio.bson", model, parameters, opt)

end


function train_video()

    model, parameters, opt = load("data/models/video.bson")

    num_files = length(readdir("data/video"))

    for (i, dir) in enumerate(readdir("data/video", join=true))

        println("$i of $num_files")

        train_over_data( model, parameters, opt, VideoIterator(dir), reshape_video )

    end

    save("data/models/video.bson", model, parameters, opt)

end