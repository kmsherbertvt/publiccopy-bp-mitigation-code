using LinearAlgebra
using Test
using AdaptBarren
using NLopt
using Random
using ProgressBars

function _run_qaoa_comparison_test(test_ham, energy_errors, max_pars, n)
    # Hyperparameters
    opt_alg = "LD_LBFGS"
    path="test_data"

    # Define pool
    pool = two_local_pool(n)
    pool = map(p -> Operator([p], [1.0]), pool)
    push!(pool, qaoa_mixer(n))

    # Initial State
    initial_state = ones(ComplexF64, 2^n) / sqrt(2^n)
    initial_state /= norm(initial_state)

    # Define Callbacks
    callbacks = Function[ParameterStopper(max_pars)]

    # Run ADAPT-QAOA
    result = adapt_qaoa(test_ham, pool, n, opt_alg, callbacks; initial_parameter=1e-2, initial_state=initial_state, path=path)

    # Define Comparison Data
    ground_state_energy = minimum(real(diag(operator_to_matrix(test_ham))))
    ho_lun_result = ADAPTHistory(
        ground_state_energy .+ energy_errors,
        [],
        [],
        [],
        [],
        [],
        [],
        []
    )

    # Compare Results
    adapt_qaoa_energy = last(result.energy)
    en_err = adapt_qaoa_energy - ground_state_energy
    @test en_err <= 1e-4 # Test that it actually gets the right result

    # Compare with Ho Lun
    function relative_err(x, y) abs(x-y)/(abs(x+y)/2) end

    errors = relative_err.(
        result.energy[2:end], # Drop the first element of mine since mine includes |+>^n
        ho_lun_result.energy
    )

    println("Comparisons")
    println("The expected errors are: $(ho_lun_result.energy)")
    println("The actual errors are:   $(result.energy[2:end])")
    println("\n")

    println("\n\n\n\n The Hamiltonian diagonal is:")
    println(real(diag(operator_to_matrix(test_ham))))
    println("\n\n\n\n")

    @test all(errors .<= 0.05)
end

@testset "Run all tests" begin

    @testset "Run Ho Lun Comparison" begin
        println("Begin Ho Lun's comparison")
        test_ham_vec = [0.0, -1.779640524849019, -0.9260288163486794, -1.9681563115877658, -1.5418617675114605, -2.6440667824502255, -2.1043764787001074, -2.469068464028941, -1.4482061210959358, -2.7084856199552063, -2.3316058016329757, -2.8543722708823145, -2.6037327216173747, -3.186576710566392, -3.123618296994383, -2.9689492563334676, -1.6430595132822896, -2.9277699712739658, -2.435395389482706, -2.98259281786445, -2.502309525394874, -3.1095844734762967, -2.931131296435258, -2.8008932149067483, -2.1243100795902614, -2.8896595115921895, -2.8740168199790386, -2.9018532223710345, -2.597224924712824, -2.685138846804498, -2.983417559941569, -2.3338184524233103, -2.3338184524233103, -2.983417559941569, -2.685138846804498, -2.597224924712824, -2.9018532223710345, -2.8740168199790386, -2.8896595115921895, -2.1243100795902614, -2.8008932149067483, -2.931131296435258, -3.1095844734762967, -2.502309525394874, -2.98259281786445, -2.435395389482706, -2.9277699712739658, -1.6430595132822896, -2.9689492563334676, -3.123618296994383, -3.186576710566392, -2.6037327216173747, -2.8543722708823145, -2.3316058016329757, -2.7084856199552063, -1.4482061210959358, -2.469068464028941, -2.1043764787001074, -2.6440667824502255, -1.5418617675114605, -1.9681563115877658, -0.9260288163486794, -1.779640524849019, 0.0]
        test_ham = matrix_to_operator(diagm(test_ham_vec))
        op_chop!(test_ham, 1e-10)
        energy_errors = [0.48591255772463215, 0.2441738332374288, 0.1535109654216682, 0.15329514239166864, 0.05175171091051345, 0.03784490678037811, 0.023216374101378445, 0.0006506746682588549, 1.1341347860849282e-08, 6.616929226765933e-14]
        _run_qaoa_comparison_test(test_ham, energy_errors, length(energy_errors), 6)
        println("\n\n\n\n\n")
    end

    @testset "Run Linghua Comparison #1" begin
        println("Begin Linghua's 8q comparison")
        n = 8
        test_edge_weights = [
            (0, 1, 0.47), 
            (1, 2, 0.05), 
            (2, 3, 0.67), 
            (3, 4, 0.28), 
            (4, 5, 0.66), 
            (5, 6, 0.21), 
            (6, 7, 0.53), 
            (0, 7, 0.86)
            ]
        # Shift index for 1-indexing
        for k=1:length(test_edge_weights)
            test_edge_weights[k] = (1, 1, 0.0) .+ test_edge_weights[k]
        end
        test_ham = max_cut_hamiltonian(n, test_edge_weights)

        energy_errors = [0.769436997845, 0.589812430072, 0.412868633019, 0.270777480032, 0.144772193615, 0.076931057079, 0.045960422324, 0.045149948093, 0.000007449742, 0.000000000063, 0.000000000001]
        _run_qaoa_comparison_test(test_ham, energy_errors, length(energy_errors), n)
        println("\n\n\n\n\n")
    end

    @testset "Run Linghua Comparison #2" begin
        println("Begin Linghua's 10q comparison")
        n = 10
        test_edge_weights = [
            (0, 1, 0.34), 
            (1, 2, 0.15), 
            (2, 3, 0.54), 
            (3, 4, 0.76), 
            (4, 5, 0.30), 
            (5, 6, 0.46), 
            (6, 7, 0.25), 
            (7, 8, 0.19), 
            (8, 9, 0.98), 
            (0, 9, 0.29)]
        # Shift index for 1-indexing
        for k=1:length(test_edge_weights)
            test_edge_weights[k] = (1, 1, 0.0) .+ test_edge_weights[k]
        end
        test_ham = max_cut_hamiltonian(n, test_edge_weights)

        energy_errors = [0.769953052898, 0.591549296798, 0.483568076593, 0.403756174889, 0.345070422803, 0.309859163300, 0.197450392784, 0.147245310891, 0.136767585688, 0.122730476287, 0.114639713897, 0.087442625608, 0.053369444337, 0.049395634541, 0.001325522123, 0.000062988891, 0.000004238123]
        _run_qaoa_comparison_test(test_ham, energy_errors, length(energy_errors), n)
        println("\n\n\n\n\n")
    end

end