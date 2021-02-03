using DataStructures

import JLD
import TensorOperations


"""
    QXContext(cmds::CommandList, params::Parameters, input_file::String, output_file::String)

A structure to represent and maintain the current state of a QXRun execution.
This structure will hold MPI rank information and is therefore responsible for
figuring out what segment of the work is its own.
"""
struct QXContext{T}
    cmds::CommandList
    params::Parameters
    input_file::String
    output_file::String
    data::Dict{Symbol, T}
end

function QXContext(::Type{T}, cmds::CommandList, params::Parameters, input_file::String, output_file::String) where {T <: AbstractArray}
    data = Dict{Symbol, T}()
    QXContext{T}(cmds, params, input_file, output_file, data)
end

function QXContext(cmds::CommandList, params::Parameters, input_file::String, output_file::String)
    QXContext(Array{ComplexF32}, cmds, params, input_file, output_file)
end

###############################################################################
# Individual command execution functions
###############################################################################

"""
    execute!(cmd::LoadCommand, ctx::ExecutionCtx{T}) where {T}

Execute a load DSL command
"""
function execute!(cmd::LoadCommand, ctx::QXContext{T}) where {T}
    ctx.data[cmd.name] = JLD.load(ctx.input_file, String(cmd.label))
    nothing
end

"""
    execute!(cmd::SaveCommand, ctx::ExecutionCtx{T}) where {T}

Execute a save DSL command
"""
function execute!(cmd::SaveCommand, ctx::QXContext{T}) where {T}
    mode = isfile(ctx.output_file) ? "r+" : "w"
    JLD.jldopen(ctx.output_file, mode) do file
        file[String(cmd.label)] = ctx.data[cmd.name]
    end
    nothing
end

"""
    execute!(cmd::DeleteCommand, ctx::ExecutionCtx{T}) where {T}

Execute a delete DSL command
"""
function execute!(cmd::DeleteCommand, ctx::QXContext{T}) where {T}
    delete!(ctx.data, cmd.label)
    nothing
end

"""
    execute!(cmd::ReshapeCommand, ctx::ExecutionCtx{T}) where {T}

Execute a reshape DSL command
"""
function execute!(cmd::ReshapeCommand, ctx::QXContext{T}) where {T}
    tensor_dims = size(ctx.data[cmd.name])
    new_dims = [prod([tensor_dims[y] for y in x]) for x in cmd.dims]
    ctx.data[cmd.name] = reshape(ctx.data[cmd.name], new_dims...)
    nothing
end

"""
    execute!(cmd::PermuteCommand, ctx::ExecutionCtx{T}) where {T}

Execute a permute DSL command
"""
function execute!(cmd::PermuteCommand, ctx::QXContext{T}) where {T}
    ctx.data[cmd.name] = permutedims(ctx.data[cmd.name], cmd.dims)
    nothing
end

"""
    execute!(cmd::NconCommand, ctx::ExecutionCtx{T}) where {T}

Execute an ncon DSL command
"""
function execute!(cmd::NconCommand, ctx::QXContext{T}) where {T}
    left_idxs = Tuple(cmd.left_idxs)
    right_idxs = Tuple(cmd.right_idxs)
    ctx.data[cmd.output_name] = TensorOperations.tensorcontract(
        ctx.data[cmd.left_name], left_idxs,
        ctx.data[cmd.right_name], right_idxs,
        Tuple(TensorOperations.symdiff(left_idxs, right_idxs))
    )
    nothing
end

"""
    execute!(cmd::ViewCommand, ctx::ExecutionCtx{T}) where {T}

Execute a view DSL command
"""
function execute!(cmd::ViewCommand, ctx::QXContext{T}) where {T}
    dims = size(ctx.data[cmd.target])
    view_index_list = [i == cmd.bond_index ? cmd.bond_range : UnitRange(1, dims[i]) for i in 1:length(dims)]
    ctx.data[cmd.name] = @view ctx.data[cmd.target][view_index_list...]
    nothing
end


###############################################################################
# Execution functions
###############################################################################

"""
    reduce_nodes(nodes::Dict{Symbol, Vector{T}})

Recombine slices results
"""
function reduce_nodes(nodes::AbstractDict{Symbol, Vector{T}}) where {T}
    sum(reduce((x,y) -> x .* y, values(nodes)))
end

"""
    execute!(ctx::QXContext{T}) where T

Run a given context.
"""
function execute!(ctx::QXContext{T}) where T
    results = Dict{String, eltype(T)}()

    input_file = JLD.jldopen(ctx.input_file, "r")

    split_idx = findfirst(x -> !(x isa LoadCommand) && !(x isa QXRun.ParametricCommand{LoadCommand}), ctx.cmds)
    # static I/O commands do not depends on any substitutions and can be run
    # once for all combinations of output qubits and slices
    static_iocmds, parametric_iocmds = begin
        iocmds = ctx.cmds[1:split_idx-1]
        pred = x -> x isa LoadCommand
        filter(pred, iocmds), filter(!pred, iocmds)
    end
    cmds = ctx.cmds[split_idx:end] 

    # Figure out the names of the tensors being loaded by iocmds
    loaded_tensors = [x.name for x in static_iocmds]
    #FIXME: This is not nice - shouldn't be exposing the ParametricCommand implementation
    append!(loaded_tensors, Symbol.([split(x.args, " ")[1] for x in parametric_iocmds]))
    # Remove any delete commands that delete tensors just loaded
    filter!(x -> !(x isa DeleteCommand && x.label in loaded_tensors), cmds)

    for iocmd in static_iocmds
        ctx.data[iocmd.name] = read(input_file, String(iocmd.label))
    end

    # run command substitution
    for substitution_set in ctx.params
        intermediate_nodes = DefaultDict{Symbol, Vector{eltype(T)}}(Vector{eltype(T)})

        # Read everything from disk together
        for iocmd in apply_substitution(parametric_iocmds, substitution_set.subs)
            ctx.data[iocmd.name] = read(input_file, String(iocmd.label))
        end

        for substitution in substitution_set
            subbed_cmds = apply_substitution(cmds, substitution)

            # Run each of the DSL commands in order
            for cmd in subbed_cmds
                if cmd isa SaveCommand
                    # The data, `ctx.data[cmd.name]`, needs to be dereferenced with `[]`
                    # Although it's a scalar, it will be within an N-d array
                    push!(intermediate_nodes[cmd.name], ctx.data[cmd.name][])
                else
                    #TODO: This could be moved out of the conditional entirely
                    #      but the symbol we're saving as would need to be updated to prevent overwrites
                    execute!(cmd, ctx)
                end
            end
        end

        results[substitution_set.amplitude] = reduce_nodes(intermediate_nodes)
    end

    close(input_file)

    #TODO: These results could also be written to `ctx.output_file`
    return results
end
