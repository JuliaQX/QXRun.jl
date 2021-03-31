using MPI
using QXRunner
using Logging

"""
    main(ARGS)

QXRunner entry point
"""
function main(args)
    if !MPI.Initialized()
        MPI.Init()
    end
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    if length(args) > 0 && args[1] === "3"
        global_logger(QXRunner.Logger.QXLoggerMPIShared())
    elseif length(args) > 0 && args[1] === "2"
        global_logger(QXRunner.Logger.QXLoggerMPIPerRank())
    else
        global_logger(QXRunner.Logger.QXLogger())
    end

    file_path      = @__DIR__
    dsl_file       = joinpath(file_path, "ghz/ghz_5.qx")
    parameter_file = joinpath(file_path, "ghz/ghz_5.yml")
    input_file     = joinpath(file_path, "ghz/ghz_5.jld2")
    output_file    = joinpath(file_path, "ghz/out.jld2")

    results = QXRunner.execute(dsl_file, parameter_file, input_file, output_file, comm)
end

main(ARGS)