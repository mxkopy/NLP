include("AutoEncoder.jl")
include("Model.jl")
include("Data.jl")

# Runs a single training iteration with backprop. 

function train_iter( model, parameters, opt, data::Array{Float32}, model_size=128 )

    # Generates an array of random floats. These are used for the reparameterization trick

    unit_gaussians = rand( Normal( 1.0, 0.1 ), model_size ) .|> Float32 |> gpu

    r_loss, d_loss = 0, 0

    gs = gradient( parameters ) do

        r_loss, d_loss = loss_function( model..., unit_gaussians, data )

        return 2 * r_loss + d_loss

    end

    Flux.Optimise.update!( opt, parameters, gs )

    return r_loss, d_loss

end



function save_model( savename, model, parameters, opt )

    serialize( savename, ( model, parameters, opt ) )

end

# This function deserializes the model and relevant parameters from the disk, 
    # and sets the data IOPipe to the checkpointed position. 

# Loading is fairly cheap, since it's done infrequently.

function load_model( savename )

    return deserialize( savename )

end



function init_model( model, savename )

    opt = ADAM( 0.01 )
    parameters = Flux.params( model... )

    save_model( savename, model, parameters, opt )

end



# we have descended into madness
function directory_iterator( data_dir, data_iterator )

    return Iterators.map( filter( x -> !occursin("checkpoint", x), readdir(data_dir, join=true) ) ) do directory

        return data_iterator( directory )

    end |> Iterators.flatten

end




function train_autoencoder( model_dir, data_dir, data_iterator, save_freq=5000 )

    model, parameters, opt   = deserialize( model_dir )

    n, r_avg, d_avg          = deserialize( data_dir * "/checkpoint" )

    directory_itr            = Iterators.drop( directory_iterator( data_dir, data_iterator ), n )

    for (i, data) in enumerate( directory_itr )

        r_loss, d_loss = train_iter( model |> gpu, parameters, opt, data |> gpu )

        r_avg, d_avg   = (r_avg + r_loss) / (n + i), (d_avg + d_loss) / (n + i)

        next_save      = (save_freq - n - i) % save_freq

        println("\rr: $r_loss d: $d_loss r_avg: $r_avg d_avg: $d_avg next_save: $next_save")

        if next_save == 0
        
            println("model saved!")
            
            serialize( model_dir, (model |> cpu, parameters, opt) )
            serialize( data_dir*"/checkpoint", (n + i, r_avg, d_avg) )

        end
        
    end


end



function save_audio( data, output )

    file = cat( data..., dims=1 )

    file = reshape(file, (size(file)[1], :))

    wavwrite(file, output, Fs=44100)

end



function save_video( data, output )

    file = map( data ) do img

        return reshape( img, (size(img)[3], size(img)[2], size(img)[1]) ) |> colorview

    end

    VideoIO.save(output, file)

end



function test_autoencoder( model_dir, data_dir, data_iterator, save_function, output, model_size=128, num_iter=10000 )

    (encoder, decoder, mean, std), _, _ = load_model( model_dir )

    encoder  = encoder[1:end-1]

    data_itr = Iterators.take( directory_iterator( data_dir, data_iterator ), num_iter )

    out = map( data_itr ) do data 

        return eval_model( encoder, decoder, mean, std, ones(model_size), data )

    end

    save_function(out, output)

end

