"""
Here we define the interface that all contexts implement and provide a simple implementation

Constructor takes compute graph and any implementation specific parameters and initialises initial
tensor and parameter storage

Each context implements the following functions to access tensor and parameter information:
gettensor(ctx, sym): get the tensor for the given symbol
settensor!(ctx, value, sym): set the tensor data for the given symbol
Base.getindex(ctx, sym): get the parameter value corresponding to given symbol
Base.setindex!(ctx, value, sym): set the parameter value corresponding to given symbol
Base.haskey(ctx, sym): Check if the parameter key exists
Base.zeros(ctx, size): create array of zeros with appropritate type for context
Base.zero(ctx): create scalar with same type used in context
Base.eltype(ctx): return the element type of numeric datastructures
set_open_bonds!(ctx, bitstring::String): Set output parameters according to provided bitstring
set_slice_vals!(ctx, slice_values::Vector{Int}): Set output parameters according to provided slice values
(c::ctx)(): Execute the compute graph in the provided context. Returns final tensor
compute_amplitude!(ctx, bitstring; num_slices=nothing): Contracts the network to compute the given bitstring. Includes reduction over slices

Also provides an implementation of each command which uses above functions to get required
tensors and parameters. For example contraction command this function is defined as

(c::ContractionCommand)(ctx): implenents the contraction described by the given contraction index with the context

Each context will also implement the following functions to be implemented by distributed contexts
ctxmap(f, ctx, items): Applies function f to items
ctxreduce(f, ctx, items): Performs reduction using the function f on items
ctxgather(ctx, items): Gathers all items
"""

using QXContexts.ComputeGraphs
using TimerOutputs
using OMEinsum
using DataStructures

export gettensor, settensor!, set_open_bonds!, set_slice_vals!
export AbstractContext, QXContext, compute_amplitude!
export ctxmap, ctxgather, ctxreduce

abstract type AbstractContext end

##################################################################################
# Provide implementation of each command
##################################################################################
"""
    (c::ContractCommand)(ctx::AbstractContext)

Execute a contraction command in the given context
"""
function (c::ContractCommand)(ctx::AbstractContext)
    output_idxs = Tuple(c.output_idxs)
    left_idxs = Tuple(c.left_idxs)
    right_idxs = Tuple(c.right_idxs)

    @timeit_debug timer_output "DSL_ncon" begin
        @debug "ncon DSL command: $(c.output_name)[$(output_idxs)] = $(c.left_name)[$(c.left_idxs)] * $(c.right_name)[$(c.right_idxs)]"
        @debug "ncon shapes: left_size=$(size(gettensor(ctx, c.left_name))), right_size=$(size(gettensor(ctx, c.right_name)))"
        settensor!(ctx, EinCode((left_idxs, right_idxs), output_idxs)(gettensor(ctx, c.left_name), gettensor(ctx, c.right_name)), c.output_name)
    end
    nothing
end

(c::LoadCommand)(ctx::AbstractContext) = settensor!(ctx, gettensor(ctx, c.label), c.name)
(c::SaveCommand)(ctx::AbstractContext) = settensor!(ctx, gettensor(ctx, c.label), c.name)

"""
    (c::ReshapeCommand)(ctx::AbstractContext)

Implementation of reshape command
"""
function (c::ReshapeCommand)(ctx::AbstractContext)
    @timeit_debug timer_output "DSL_reshape" begin
        tensor_dims = size(gettensor(ctx, c.input))
        new_dims = [prod([tensor_dims[y] for y in x]) for x in c.dims]
        settensor!(ctx, reshape(gettensor(ctx, c.input), new_dims...), c.output)
    @debug "Reshape DSL command: name=$(c.input)($(tensor_dims)) -> $(c.output)($(new_dims))"
    end
    nothing
end

"""
    (c::ViewCommand)(ctx::AbstractContext)

Execute the given view command using provided context
"""
function (c::ViewCommand)(ctx::AbstractContext)
    @timeit_debug timer_output "DSL_view" begin
        bond_val = haskey(ctx, c.slice_sym) ? ctx[c.slice_sym] : nothing
        if bond_val !== nothing
            dims = size(gettensor(ctx, c.input_sym))
            view_index_list = [i == c.bond_index ? UnitRange(bond_val, bond_val) : UnitRange(1, dims[i]) for i in 1:length(dims)]
            new_tensor = @view gettensor(ctx, c.input_sym)[view_index_list...]
            settensor!(ctx, new_tensor, c.output_sym)
            @debug "view DSL command: $(c.output_sym) = $(c.input_sym)[$(view_index_list)]"
        else
            settensor!(ctx, gettensor(ctx, c.input_sym), c.output_sym)
            @debug "view DSL command: $(c.output_sym) = $(c.input_sym)"
        end
    end
    nothing
end

"""
    (c::OutputCommand)(ctx::AbstractContext)

Execute the given output command using provided context
"""
function (c::OutputCommand)(ctx::AbstractContext)
    sym = Symbol("o$(c.idx)")
    @assert haskey(ctx, sym) "Output $sym not set in context"
    out_val = ctx[sym]
    data_array = zeros(ctx, c.dim)
    data_array[out_val + 1] = 1.
    settensor!(ctx, data_array, c.name)
    @debug "$(c.name) = $(data_array)"
end

##########################################################################################
# Provide concreate implementation of a context
#########################################################################################

"""Data structure for context implementation"""
struct QXContext{T} <: AbstractContext
    params::Dict{Symbol, Int}
    tensors::Dict{Symbol, T}
    cg::ComputeGraph
    slice_dims::OrderedDict{Symbol, Int}
    output_dims::OrderedDict{Symbol, Int}
end

"""
    QXContext{T}(cg::ComputeGraph) where T

Constuctor which initialises instance from a compute graph
"""
function QXContext{T}(cg::ComputeGraph) where T
    tensors = Dict{Symbol, T}()
    # load tensors from compute graph into this dictionary
    for (k, v) in pairs(cg.tensors)
        tensors[k] = convert(T, v)
    end
    slice_dims = convert(OrderedDict, params(cg, ViewCommand))
    output_dims = convert(OrderedDict, params(cg, OutputCommand))
    sort!(slice_dims)
    sort!(output_dims)
    QXContext{T}(Dict{Symbol, Int}(), tensors, cg, slice_dims, output_dims)
end

QXContext(cg::ComputeGraph) = QXContext{Array{ComplexF32}}(cg)

"""
    gettensor(ctx::QXContext, sym)

Function to retrieve tensors by key
"""
function gettensor(ctx::QXContext, sym)
    @assert haskey(ctx.tensors, sym) "Tensor $sym does not exist in this context"
    ctx.tensors[sym]
end

"""
    settensor!(ctx::QXContext, sym)

Function to set tensors by key
"""
function settensor!(ctx::QXContext, value, sym)
    ctx.tensors[sym] = value
end

"""Implement has key to check if parameter by this name present"""
Base.haskey(ctx::QXContext, sym) = haskey(ctx.params, sym)

"""
    Base.getindex(ctx::QXContext, sym)::Int

Implement getindex for retrieving parameter values by key
"""
function Base.getindex(ctx::QXContext, sym)::Int
    @assert haskey(ctx, sym) "Parameter $sym does not exist in this context"
    ctx.params[sym]
end

"""
    Base.setindex!(ctx::QXContext, sym)::Int

Implement setindex! for setting parameter values
"""
function Base.setindex!(ctx::QXContext, value, sym)
    ctx.params[sym] = value
end

Base.zeros(::QXContext{T}, size) where T = zeros(eltype(T), size)
Base.zero(::QXContext{T}) where T = zero(eltype(T))
Base.eltype(::QXContext{T}) where T = eltype(T)

"""
    set_open_bonds!(ctx::QXContext, bitstring::String)

Given a bitstring, set the open bonds to values so contracting the network will
calculate the amplitude of this bitstring
"""
function set_open_bonds!(ctx::QXContext, bitstring::String)
    @assert length(bitstring) == length(ctx.output_dims) "Bitstring length must match nubmer of outputs"
    for (i, key) in enumerate(keys(ctx.output_dims)) ctx[key] = parse(Int, bitstring[i]) end
end

"""
    set_slice_vals!(ctx::QXContext, slice_values::Vector{Int})

For each bond that is being sliced set the dimension to slice on.
"""
function set_slice_vals!(ctx::QXContext, slice_values::Vector{Int})
    # set slice values and remove any already set
    for (i, key) in enumerate(keys(ctx.slice_dims))
        if i > length(slice_values)
            delete!(ctx.params, key)
        else
            ctx[key] = slice_values[i]
        end
    end
end

"""
    (ctx::QXContext)()

Funciton to execute compute graph when struct is called. Returns final tensor
"""
function (ctx::QXContext)()
    for n in get_commands(ctx.cg) n(ctx) end
    gettensor(ctx, output(ctx.cg))
end

"""
    compute_amplitude!(ctx::QXContext, bitstring::String; max_slices=nothing)

Calculate a single amplitude with the given context and bitstring. Involves a sum over
contributions from each slice. Can optionally set the number of bonds. By default all slices
are used.
"""
function compute_amplitude!(ctx::QXContext, bitstring::String; max_slices=nothing)
    set_open_bonds!(ctx, bitstring)
    amplitude = nothing
    for p in SliceIterator(ctx.cg, max_slices=max_slices)
        set_slice_vals!(ctx, p)
        if amplitude === nothing
            amplitude = ctx()
        else
            amplitude += ctx()
        end
    end
    amplitude = convert(Array, amplitude) # if a gpu array convert back to gpu
    if ndims(amplitude) == 0
        amplitude = amplitude[]
    end
    amplitude
end

"""Map over items as placeholder for more complicated contexts"""
ctxmap(f, ctx::QXContext, items) = map(f, items)

"""Simple gather as placeholder for distributed contexts"""
ctxgather(ctx::QXContext, items) = items

"""Simple gather as placeholder for distributed contexts"""
ctxreduce(f, ctx::QXContext, items) = reduce(f, items)