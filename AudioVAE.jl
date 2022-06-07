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



function audio_encoder(model_size, audio_size)

    return Chain(

        DenseFFT( audio_size, 512, bias=false ),
        BatchNorm(2),
        DenseFFT(512, model_size, sigmoid, bias=true),
        Dropout(0.5),
 
    )

end


function audio_decoder(model_size, audio_size )

    return Chain(

        Dropout(0.5),
        DenseFFT( model_size, 512, sigmoid, bias=true ),
        DenseFFT( 512, audio_size, bias=false )

    )

end




export init_complex_array, DenseFFT, audio_encoder, audio_decoder

end