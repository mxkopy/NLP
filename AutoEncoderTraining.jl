include("AutoEncoders.jl")
include("Data.jl")

using Printf
using BSON: @save, @load
using Zygote: @ignore
using Flux: @epochs


function loss_function( model::AutoEncoder, p, data )

    enc_out, means, devs, latent, dec_out = model(p, data)

    r_loss = Flux.Losses.mse( dec_out, data )

    d_loss = Flux.Losses.kldivergence( softmax(devs), softmax(means) )

    Zygote.ignore() do 
        @printf "\rr_loss %.5e d_loss %.5e" r_loss d_loss 
        flush(stdout)
    end

    return r_loss + 0.5 * d_loss

end


function save_model( savename, model, optimizer )

    @save savename model optimizer
    GC.gc(true)
    CUDA.reclaim()

end


function train_autoencoder( model, optimizer, data_iterator, savename; save_freq=10, epochs=1 )

    trainmode!(model)

    model      = model |> data_iterator.device

    loss       = (p, data) -> loss_function( model, p, data )

    callback   = Flux.throttle( () -> save_model( savename, model, optimizer ), save_freq )

    parameters = Flux.params( model )

    @epochs epochs Flux.Optimise.train!( loss, parameters, data_iterator, optimizer, cb=callback )

end



function save_audio( data, output )

    file = reduce( (l, r) -> cat( l, r, dims=1 ), data )

    file = reshape(file, (size(file)[1], :))

    wavwrite(file, output, Fs=44100)

end



function save_video( data, output )

    file = map( data ) do img

        img = reshape( img, (size(img)[3], size(img)[2], size(img)[1]) ) |> unit_denormalize
        img = clamp.( img, N0f8 )
        img = colorview(RGB, img)

        return img

    end

    VideoIO.save(output, file)

end



function test_autoencoder( model, data_iterator, save_function, output, num_iter )

    testmode!( model )

    model = model |> data_iterator.device

    data = mapreduce( (l, r) -> cat( l, r, dims=1 ), Iterators.take( data_iterator, num_iter )) do (param, data)

        _, _, _, _, y = model( param, data )

        return [y |> cpu]

    end

    save_function(data, output)

end

