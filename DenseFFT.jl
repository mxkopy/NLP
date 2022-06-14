using Flux, FFTW, SliceMap, CUDA

struct DenseFFT

    W::AbstractArray{<:Complex}
    b::Union{AbstractArray{<:Complex}, Bool}
    a::Function

end


# https://github.com/FluxML/Zygote.jl/issues/1121

gpu(x::DenseFFT) = x |> cpu

function (layer::DenseFFT)(data::AbstractArray{<:Real})

    out = layer.a.( data |> cpu )

    out = FFTW.fft( out, 1 )
    out = reshape(out, ( 1, size(out)[1], : ) )

    out = NNlib.batched_mul(out, layer.W)
    out = reshape(out, (size(out)[2], :)) .+ layer.b

    out = reshape(out, (:, size(data)[2:end]...) )
    out = FFTW.ifft(out, 1) .|> real

    return typeof(data) <: CuArray ? out |> gpu : out

end


init_complex_array( shape, init_real=Flux.glorot_uniform, init_imag=Flux.glorot_uniform ) = Complex.( init_real( reduce(*, shape) ), init_imag( reduce(*, shape) ) ) |> x -> reshape( x, shape )

DenseFFT( dim_in::Int, dim_out::Int, activation=identity; init=Flux.glorot_uniform, bias=true ) = DenseFFT( init_complex_array( ( dim_in, dim_out ), init ), bias ? init_complex_array( dim_out, init ) : false, activation ) 

Flux.@functor DenseFFT 
# Flux.trainable(c::DenseFFT) = (c.W, c.b, c.a)
