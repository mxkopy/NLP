include("Words.jl")
include("Transformers.jl")

using Flux, .Transformers
using BSON: @save, @load

# assume input has an end token appended
function loss_function( transformer::Transformer, input, target, previous_output )

    output = transformer( input, previous_output )

    pad    = zeros( max(0, size(output)[1] - size(target)[1] ) )

    error  = Flux.logitcrossentropy( cat(target, pad, dims=1)[1:size(output)[1], :], output )

    return error

end



function train_transformer( model, optimizer, data_iterator, savename; save_freq=10, epochs=8 )


end