module AudioVAE

using Flux, FFTW, SliceMap


struct DenseFFT

    W::Array{<:Complex}
    b::Array{<:Complex}

end


function dense_fft( layer::DenseFFT, data::AbstractArray{<: Real, 1} )

    x   = FFTW.fft( data )
    frq = ( transpose( layer.W ) * x ) .+ layer.b
    out = FFTW.ifft( frq ) .|> real

    return out

end


function (layer::DenseFFT)(data::Array{<: Real})

    return slicemap( x -> dense_fft( layer, x ), data, dims=1 )

end


init_complex_array( shape, init_real=Flux.glorot_normal, init_imag=init_real ) = Complex.( init_real( reduce(*, shape) ), init_imag( reduce(*, shape) ) ) |> x -> reshape( x, shape )

DenseFFT( dim_in::Int, dim_out::Int; init=Flux.glorot_normal ) = DenseFFT( init_complex_array( ( dim_in, dim_out ), init ), init_complex_array( dim_out, init ) )

Flux.@functor DenseFFT

gpu( x::DenseFFT ) = DenseFFT( x.W |> gpu, x.b |> gpu )


function coder_conv( in_channels, out_channels, kernel, conv_type )

    return Chain(
        
        conv_type( kernel, in_channels => out_channels, pad=SamePad() ),
        BatchNorm( out_channels ),
        x -> relu.(x)
    )

end



function coder_block( in_channels, out_channels, input_size, output_size, conv_type )

    return Chain( 
        
        DenseFFT(input_size, output_size), 
        coder_conv( in_channels, out_channels, (1, 1), conv_type )...

    )

end


function audio_coder(conv_type, channels, sizes )

    Chain( [ coder_block( channels[i], channels[i+1], sizes[i], sizes[i+1], conv_type ) for i in 1:length(channels)-1 ]... )

end


function audio_encoder(model_size, audio_size; channels=[2, 32, 64, model_size], sizes=[audio_size, 220, 32, 1])

    return audio_coder( Conv, channels, sizes )

end


function audio_decoder(model_size, audio_size; channels=[model_size, 64, 32, 2], sizes=[1, 32, 220, audio_size])

    return audio_coder( ConvTranspose, channels, sizes )

end




export init_complex_array, DenseFFT, audio_encoder, audio_decoder

end