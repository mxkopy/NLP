module AudioVAE

using Flux, FFTW, SliceMap

struct DenseFFT

    W::AbstractArray{<:Complex}
    b::AbstractArray{<:Complex}

end



function (layer::DenseFFT)(data::AbstractArray{<:Real})

    out = FFTW.fft( data, 1 )
    out = reshape(out, ( 1, size(out)[1], : ) )

    out = NNlib.batched_mul(out, layer.W)
    out = reshape(out, (size(out)[2], :)) .+ layer.b

    out = reshape(out, (length(layer.b), size(data)[2:end]...) )
    out = FFTW.ifft(out, 1) .|> real

    return out

end



init_complex_array( shape, init_real=Flux.glorot_uniform, init_imag=Flux.glorot_uniform ) = Complex.( init_real( reduce(*, shape) ), init_imag( reduce(*, shape) ) ) |> x -> reshape( x, shape )

DenseFFT( dim_in::Int, dim_out::Int; init=Flux.glorot_uniform ) = DenseFFT( init_complex_array( ( dim_in, dim_out ), init ), init_complex_array( dim_out, init ) )

Flux.@functor DenseFFT



function coder_block(  input_size, output_size )

    return Chain( 

        DenseFFT(input_size, output_size), 
        Dropout(0.3)

    )

end


function audio_coder( sizes )

    Chain( [ coder_block( sizes[i], sizes[i+1] ) for i in 1:length(sizes)-1 ]... )

end


function audio_encoder(model_size, audio_size; sizes=[audio_size, model_size] )

    return audio_coder( sizes )

end


function audio_decoder(model_size, audio_size; sizes=[model_size, audio_size] )

    return audio_coder( sizes )

end




export init_complex_array, DenseFFT, audio_encoder, audio_decoder

end