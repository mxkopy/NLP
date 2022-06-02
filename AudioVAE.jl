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


IFFTKernel(size; init=Flux.glorot_normal) = IFFTKernel( init( (data_size, channels) ) )


Flux.@functor FFT_Kernel


function audio_coder( input_size, in_channels, out_channels, kernel, stride, conv_type )

    return Chain(

        FFTW.fft, 
        IFFTKernel(input_size), 
        conv_type( (kernel, kernel), in_channels=>out_channels, stride=stride, pad=SamePad() )

    )

end



