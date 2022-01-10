using Flux, FFTW, SliceMap



struct IFFTKernel

    # Kernel shape should be (size, 1)
    kernel::Array{ComplexF32}

end


function (ifft::IFFTKernel)(data::Array{ComplexF32})

    frequencies = ifft.kernel .* data
    out         = FFTW.ifft( frequencies )

    return out

end


IFFTKernel(size; init=Flux.glorot_normal) = IFFTKernel( init( size ) )


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