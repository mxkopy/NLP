using Flux, FFTW, SliceMap


struct DenseFFT

    W::Array{Complex}
    b::Array{Complex}

end


function dense_ifft( layer::DenseFFT, data::AbstractArray{<: Real, 1} )

    x   = FFTW.fft( data )
    frq = ( transpose( layer.W ) * x ) .+ layer.b
    out = FFTW.ifft( frq ) .|> real

    return out

end


function (layer::DenseFFT)(data::Array{<: Real} )

    return slicemap( x -> dense_ifft( layer, x ), data, dims=1 )

end


init_complex_array( shape, init ) = Complex.( init( reduce(*, shape) ), init( reduce(*, shape) ) ) |> x -> reshape( x, shape )


DenseFFT( dim_in::Int, dim_out::Int; init=Flux.glorot_normal ) = DenseFFT( init_complex_array( ( dim_in, dim_out ), init ), init_complex_array( dim_out, init ) )


Flux.@functor DenseFFT







function audio_coder( input_size, in_channels, out_channels, kernel, stride, conv_type )

    return Chain(

        DenseFFT( input_size, input_size ), 
        conv_type( (kernel, kernel), in_channels=>out_channels, stride=stride, pad=SamePad() )

    )

end


function audio_encoder(;input_size=1764, model_size=128, channels = [2, 16, 64, 128], kernels=[49, 9, 4] )

    layers = map( 1:length(kernels) ) do i

        return audio_coder( input_size ÷ reduce(*, kernels[1:i]), channels[i], channels[i+1], kernels[i], kernels[i], Conv )

    end

    return Chain( layers... )

end