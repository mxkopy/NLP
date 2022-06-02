using FileIO, VideoIO, Images, WAV, CUDA


# data preprocessing
reshape_audio =  x -> reshape(Float32.(x), (size(x)[1], 1, size(x)[2], :) )
reshape_video =  x -> reshape(Float32.(x), (size(x)[3], size(x)[2], size(x)[1], :) )


function image_to_array( image )

    return image |> channelview |> reshape_video

end


function video_iterator( dir, image_size )

    return Iterators.map( x -> imresize(x, (image_size, image_size) ) |> image_to_array, VideoIO.load( dir ) )

end



function replace_runoff( buffer, sample_size )

    runoff_size = size(buffer)[1] % sample_size
    temp        = zeros((sample_size - runoff_size, size(buffer)[2]))
    buffer      = cat(buffer, temp, dims=1)

    return buffer

end



function reshape_file( buffer, sample_size )

    return reshape( buffer, (sample_size, size(buffer)[2], size(buffer)[1] ÷ sample_size ) )

end



function audio_iterator( dir, sample_size )

    curr_file, _ = wavread( dir )
    curr_file    = replace_runoff( curr_file, sample_size )
    curr_file    = reshape_file( curr_file, sample_size )

    return Iterators.map( reshape_audio, eachslice(curr_file, dims=3) )

end



# we have descended into madness
function directory_iterator( data_dir, data_iterator, args... )

    return Iterators.map( filter( x -> !occursin(".bson", x), readdir(data_dir, join=true) ) ) do directory

        return data_iterator( directory, args... )

    end |> Iterators.flatten

end



struct BatchIterator

    itr
    batches::Int

end


AudioIterator( dir, sample_size; batches=4 ) = BatchIterator( directory_iterator( dir, audio_iterator, sample_size ) |> Iterators.Stateful, batches )
VideoIterator( dir, image_size; batches=4 )  = BatchIterator( directory_iterator( dir, video_iterator, image_size )  |> Iterators.Stateful, batches )


function Base.iterate( itr::BatchIterator, state=itr.itr )

    return isempty( state ) ? nothing : ( reduce( (l, r) -> cat(l, r, dims=4), Iterators.take(state, itr.batches) ), itr.itr )

end




