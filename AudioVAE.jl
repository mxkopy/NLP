using Flux, FFTW, SliceMap


struct DenseIFFT

    W::Array{Complex}
    b::Array{Complex}

end


function dense_ifft( layer::DenseIFFT, x::Vector{Complex} )

    frq = ( x * layer.W ) .+ layer.b
    out = FFTW.ifft( frq ) .|> real

end


function (layer::DenseIFFT)(data::Array{Complex})

    return slicemap( x -> dense_ifft( layer, x ), data, dims=1 )

end


init_complex_array( shape, init ) = Complex.( init( reduce(*, shape) ), init( reduce(*, shape) ) ) |> x -> reshape(x, shape )


DenseIFFT( dim_in, dim_out; init=Flux.glorot_normal ) = DenseIFFT( init_complex_array( ( dim_in, dim_out ), init ), init_complex_array( dim_out, init ) )


Flux.@functor IFFTKernel


function audio_coder( input_size, in_channels, out_channels, kernel, stride, conv_type )

    return Chain(

        FFTW.fft, 
        IFFTKernel( ( input_size, 1 ) ), 
        conv_type( (kernel, kernel), in_channels=>out_channels, stride=stride, pad=SamePad() )

    )

end


function audio_encoder(;input_size=1764, model_size=128, channels = [2, 16, 64, 128], kernels=[49, 9, 4] )

    layers = map( 1:length(kernels) ) do i

        return audio_coder( input_size ÷ reduce(*, kernels[1:i]), channels[i], channels[i+1], kernels[i], kernels[i], Conv )

    end

    return Chain( layers... )

end