module Sampling

export ListSampler, RejectionSampler, UniformSampler
export create_sampler

using Random
using DataStructures

using QXContexts.Contexts

# Module containing sampler objects which provide different levels of sampling features.
# Each sampler has a constructor which takes a context to perform sampling in and a set
# of keyword arguments that control the sampling behavior.
#
# Sampler(ctx; kwargs...): Initialise the sampler
#
# Each sampler is also callable with arguments that control it's execution
#
# (s::Sampler)(kwargs...): Perform sampling and return sampling results
#

"""Abstract type for samplers"""
abstract type AbstractSampler end

"""Functions to generate random bitstrings"""
random_bitstring(rng, num_qubits) = prod(rand(rng, ["0", "1"], num_qubits))
random_bitstrings(rng, num_qubits, num_samples) = [random_bitstring(rng, num_qubits) for _ in 1:num_samples]

###############################################################################
# ListSampler
###############################################################################

"""
A Sampler struct to compute the amplitudes for a list of bitstrings.
"""
struct ListSampler <: AbstractSampler
    ctx::AbstractContext
    list::Vector{String}
end

"""
    ListSampler(ctx
                ;bitstrings::Vector{String}=String[],
                rank::Integer=0,
                comm_size::Integer=1,
                kwargs...)

Constructor for a ListSampler to produce a portion of the given `bitstrings` determined by
the given `rank` and `comm_size`.
"""
function ListSampler(ctx
                     ;bitstrings::Vector{String}=String[],
                     kwargs...)
    if haskey(kwargs, :num_samples)
        n = kwargs[:num_samples]
        n = min(n, length(bitstrings))
    else
        n = length(bitstrings)
    end

    ListSampler(ctx, bitstrings[1:n])
end

"""
    (s::ListSampler)(max_amplitudes=nothing, kwargs...)

Callable for ListSampler struct. Calculates amplitudes for each bitstring in the list
"""
function (s::ListSampler)(;max_amplitudes=nothing, kwargs...)
    bs = if max_amplitudes === nothing
        s.list
    else s.list[1:min(max_amplitudes, length(s.list))] end

    amps = ctxmap(s.ctx, x -> compute_amplitude!(s.ctx, x; kwargs...), bs)
    amps = ctxgather(s.ctx, amps)
    (bs, amps)
end

create_sampler(ctx, sampler_params) = get_constructor(sampler_params[:method])(ctx ;sampler_params[:params]...)
get_constructor(func_name::String) = getfield(Main, Symbol(func_name*"Sampler"))


###############################################################################
# RejectionSampler
###############################################################################

"""
A Sampler struct to use rejection sampling to produce output.
"""
mutable struct RejectionSampler <: AbstractSampler
    ctx::AbstractContext
    num_qubits::Integer
    num_samples::Integer
    M::Real
    fix_M::Bool
    rng::MersenneTwister
end

"""
    function RejectionSampler(;num_qubits::Integer,
                              num_samples::Integer,
                              M::Real=0.0001,
                              fix_M::Bool=false,
                              seed::Integer=42,
                              kwargs...)

Constructor for a RejectionSampler to produce and accept a number of bitstrings.
"""
function RejectionSampler(ctx::AbstractContext;
                          num_qubits::Integer,
                          num_samples::Integer,
                          M::Real=0.0001,
                          fix_M::Bool=false,
                          seed::Integer=42,
                          kwargs...)
    # Evenly divide the number of bitstrings to be sampled amongst the subgroups of ranks.
    # num_samples = get_rank_size(num_samples, comm_size, rank)
    rng = MersenneTwister(seed) # TODO: should somehow add the rank to the seed, maybe with get_rank(ctx)?
    RejectionSampler(ctx, num_qubits, num_samples, M, fix_M, rng)
end

"""
    (s::RejectionSampler)(max_amplitudes=nothing, kwargs...)

Callable for RejectionSampler struct. Computes amplitudes for uniformly distributed bitstrings and corrects the distribution
using a rejection step.
"""
function (s::RejectionSampler)(;max_amplitudes=nothing, kwargs...)
    num_samples = max_amplitudes === nothing ? s.num_samples : max_amplitudes
    samples = Array{String, 1}(undef, num_samples)
    N = 2^s.num_qubits

    accepted = 0
    while accepted < num_samples
        # produce cadidate bitstrings
        bitstrings = random_bitstrings(s.rng, s.num_qubits, num_samples-accepted)

        # compute amplitudes for the bitstrings
        amps = ctxmap(s.ctx, x -> compute_amplitude!(s.ctx, x; kwargs...), bitstrings)
        amps = ctxgather(s.ctx, amps)

        # Perform a rejection step for each bitstring to correct the distribution of samples.
        for (bs, amp) in zip(bitstrings, amps)
            Np = N * abs(amp)^2
            s.fix_M && (s.M = max(Np, s.M))

            # Rejection step.
            if rand(s.rng) < Np / s.M
                accepted += 1
                samples[accepted] = bs
            end
        end
    end

    samples
end


###############################################################################
# UniformSampler
###############################################################################

"""
A Sampler struct to uniformly sample bitstrings and compute their amplitudes.
"""
mutable struct UniformSampler <: AbstractSampler
    ctx::AbstractContext
    num_qubits::Integer
    num_samples::Integer
    rng::MersenneTwister
end

"""
    UniformSampler(ctx::AbstractContext;
                    num_qubits::Integer,
                    num_samples::Integer,
                    seed::Integer=42,
                    kwargs...)

Constructor for a UniformSampler to uniformly sample bitstrings.
"""
function UniformSampler(ctx::AbstractContext;
                        num_qubits::Integer,
                        num_samples::Integer,
                        seed::Integer=42,
                        kwargs...)
    # Evenly divide the number of bitstrings to be sampled amongst the subgroups of ranks.
    # num_samples = (num_samples ÷ comm_size) + (rank < num_samples % comm_size)
    rng = MersenneTwister(seed)
    UniformSampler(ctx, num_qubits, num_samples, rng)
end

"""
    (s::UniformSampler)(max_amplitudes=nothing, kwargs...)

Callable for UniformSampler struct. Computes amplitudes for uniformly distributed bitstrings.
"""
function (s::UniformSampler)(;max_amplitudes=nothing, kwargs...)
    num_samples = max_amplitudes === nothing ? s.num_samples : max_amplitudes

    bs = random_bitstrings(s.rng, s.num_qubits, num_samples)
    amps = ctxmap(s.ctx, x -> compute_amplitude!(s.ctx, x; kwargs...), bs)
    amps = ctxgather(s.ctx, amps)

    (bs, amps)
end

# ###############################################################################
# # Sampler Constructor
# ###############################################################################

# """
#     create_sampler(params)

# Returns a sampler whose type and parameters are specified in the Dict `params`.

# Additional parameters that determine load balancing and totale amout of work to be done
# are set by `max_amplitudes` and the Context `ctx`.
# """
# function create_sampler(params, ctx, max_amplitudes=nothing)
#     max_amplitudes === nothing || (params[:params][:num_samples] = max_amplitudes)
#     create_sampler(params, ctx)
# end

# function create_sampler(params, ctx::QXMPIContext)
#     params[:rank] = MPI.Comm_rank(ctx.comm) ÷ MPI.Comm_size(ctx.sub_comm)
#     params[:comm_size] = MPI.Comm_size(ctx.comm) ÷ MPI.Comm_size(ctx.sub_comm)
#     create_sampler(params)
# end


# """
# Struct to hold the results of a simulation.
# """
# struct Samples{T}
#     bitstrings_counts::DefaultDict{String, <:Integer}
#     amplitudes::Dict{String, T}
# end

# Samples() = Samples(DefaultDict{String, Int}(0), Dict{String, ComplexF32}())

# """Function for reducing over amplitude contributions from each slice. For non-serial
# contexts, contributions are summed over"""
# reduce_slices(::QXContext, a) = a

# """Function for reducing over calculated amplitudes and samples. For non-serial contexts,
# contributions are gathered"""
# reduce_results(::QXContext, results::Samples) = results

# """Function for reducing over calculated amplitudes. For non-serial contexts, contributions
# are gathered"""
# BitstringIterator(::QXContext, bitstrings) = bitstrings

end