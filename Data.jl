using FileIO, VideoIO, Images, WAV, CUDA

# all video will be at 25 fps
# all audio will have a samplerate of 44100

function image_to_array( image )

    return Float32.( image |> channelview )

end


function load_video_iterator( dir )

    return Iterators.map( image_to_array, VideoIO.load( dir ) )

end

VideoIterator( dir ) = load_video_iterator( dir )



function replace_runoff( buffer, sample_size )

    runoff_size = size(buffer)[1] % sample_size
    temp        = zeros((sample_size - runoff_size, size(buffer)[2]))
    buffer      = cat(buffer, temp, dims=1)

    return buffer

end



function reshape_file( buffer, sample_size )

    return reshape( buffer, (sample_size, size(buffer)[2], size(buffer)[1] ÷ sample_size ) )

end



function load_audio_iterator( dir, sample_size=44100÷25 )

    curr_file, _ = wavread( dir )
    curr_file    = replace_runoff( curr_file, sample_size )
    curr_file    = reshape_file( curr_file, sample_size )

    return eachslice(curr_file, dims=3) 

end

AudioIterator( dir ) = load_audio_iterator( dir )
