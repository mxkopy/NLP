module AudioVAE

using Flux, FFTW, SliceMap

struct DenseFFT

    W::AbstractArray{<:Complex}
    b::Union{AbstractArray{<:Complex}, Bool}
    a::Function

end



function (layer::DenseFFT)(data::AbstractArray{<:Real})

    out = layer.a.( data )
    out = FFTW.fft( out, 1 )
    out = reshape(out, ( 1, size(out)[1], : ) )

    out = NNlib.batched_mul(out, layer.W)
    out = reshape(out, (size(out)[2], :)) .+ layer.b

    out = reshape(out, (:, size(data)[2:end]...) )
    out = FFTW.ifft(out, 1) .|> real

    return out

end


init_complex_array( shape, init_real=Flux.glorot_uniform, init_imag=Flux.glorot_uniform ) = Complex.( init_real( reduce(*, shape) ), init_imag( reduce(*, shape) ) ) |> x -> reshape( x, shape )

DenseFFT( dim_in::Int, dim_out::Int, activation=identity; init=Flux.glorot_uniform, bias=true ) = DenseFFT( init_complex_array( ( dim_in, dim_out ), init ), bias ? init_complex_array( dim_out, init ) : false, activation ) 

Flux.@functor DenseFFT


function residual_fft_block( output_size )

    return SkipConnection( DenseFFT( output_size, output_size ), + )

end


function audio_encoder(model_size, audio_size)

    return Chain(

        Conv( (3, 1), 2 => 4,   stride=1 ),
        Conv( (5, 1), 4 => 8,   stride=2 ),
        Conv( (5, 1), 8 => 16,  stride=2 ),
        Conv( (5, 1), 16 => 32, stride=2 ),
        AdaptiveMeanPool( (model_size, 1)),
        # DenseFFT( model_size, model_size )
 
    )

end


function audio_decoder(model_size, audio_size )

    return Chain(

        # DenseFFT( model_size, model_size, bias=false ),
        ConvTranspose( (5, 1),   32 => 16, stride=2 ),
        ConvTranspose( (5, 1),   16 => 8,  stride=2 ),
        ConvTranspose( (5, 1),   8 => 4,   stride=2 ),
        ConvTranspose( (5, 1),   4 => 4,   stride=2 ),
        ConvTranspose( (3, 1),   4 => 2,   stride=1 )
    )

end




export init_complex_array, DenseFFT, audio_encoder, audio_decoder

end