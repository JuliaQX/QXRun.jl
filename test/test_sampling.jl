module SamplingTests

using FileIO
using Test


using QXContexts

include("utils.jl")

@testset "Sampling tests" begin

    test_path = dirname(@__DIR__)
    dsl_file = joinpath(test_path, "examples/ghz/ghz_5.qx")
    input_data_file = joinpath(test_path, "examples/ghz/ghz_5.jld2")
    param_file  = joinpath(test_path, "examples/ghz/ghz_5.yml")

    mktempdir() do path
        output_data_file = joinpath(path, "out.jld2")
        execute(dsl_file, param_file, input_data_file, output_data_file)

        # ensure all dictionary entries match
        output = load(output_data_file, "results")
        @test output[1] == collect(keys(ghz_results))
        @test output[2] ≈ collect(values(ghz_results))
    end

    # Test rejection sampling
    param_file = joinpath(test_path, "examples/ghz/ghz_5_rejection.yml")

    mktempdir() do path
        output_data_file = joinpath(path, "out.jld2")
        execute(dsl_file, param_file, input_data_file, output_data_file)

        # ensure all dictionary entries match
        output = FileIO.load(output_data_file, "results")
        @test length(output) == 10 # Should only have 10 samples
        @test length(unique(output)) == 2 # output should only contain strings "11111" and "00000"
    end

    # Test uniform sampling
    param_file = joinpath(test_path, "examples/ghz/ghz_5_uniform.yml")

    mktempdir() do path
        output_data_file = joinpath(path, "out.jld2")
        execute(dsl_file, param_file, input_data_file, output_data_file)

        # ensure all dictionary entries match
        output = FileIO.load(output_data_file, "results")
        @test length(output[1]) == 10 # Should only have 10 samples
        @test typeof(output[1][1]) == String
        @test length(output[2]) == 10 # should only have 10 amplitudes
        @test typeof(output[2][1]) <: Complex
    end
end

end