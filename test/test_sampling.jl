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
        output = FileIO.load(output_data_file, "bitstrings_counts")
        @test length(output) == 2
        @test output["11111"] + output["00000"] == 10
    end
end

end