export runclassifier, testclassifier, buildmodel, classifierparams, classifier_param_names,
    LassoPathClassifiers, LassoClassifier

import StatsModels: DummyCoding, FullDummyCoding
export DummyCoding, FullDummyCoding

# const __classifierparams__ = (
#     svm_radial        = (:C,:gamma),
#     svm_linear        = (:C,),
#     gradient_boosting = (:max_depth, :n_estimators, :learning_rate),
#     logistic_l1       = (:lambda,),
# )
# classifier_param_names(classifier) = __classifierparams__[classifier]
# function classifierparams(obj, classifier)
#     (;(p => obj[p] for p in __classifierparams__[classifier])...)
# end

struct LassoClassifier
    lambda::Float64
end
struct LassoClassifierFit{T}
    result::T
    μ::Array{Float64,2}
    σ::Array{Float64,2}
end

zscorecol(X, μ, σ) = zscorecol!(copy(X), μ, σ)
function zscorecol!(X, μ, σ)
    for col in 1:size(X,2)
        X[:,col] .= zscore(view(X,:,col), view(μ, :, col), view(σ, :, col))
    end
    X
end

function ScikitLearn.fit!(model::LassoClassifier, X, y; kwds...)
    μ = mean(X, dims = 1)
    σ = std(X, dims = 1)

    fit = StatsBase.fit(LassoModel, zscorecol(X, μ, σ), y,
        Bernoulli(), λ = [model.lambda], standardize = false; kwds...)
    LassoClassifierFit(fit, μ, σ)
end
ScikitLearn.predict(model::LassoClassifierFit, X) =
    StatsBase.predict(model.result, zscorecol(X, model.μ, model.σ))

struct LassoPathClassifiers
    lambdas::Vector{Float64}
end
struct LassoPathFits{T}
    result::T
    μ::Array{Float64,2}
    σ::Array{Float64,2}
end
function ScikitLearn.fit!(model::LassoPathClassifiers, X, y; kwds...)
    μ = mean(X, dims = 1)
    σ = std(X, dims = 1)

    fits = StatsBase.fit(LassoPath, zscorecol(X, μ, σ), y,
        Bernoulli(), λ = model.lambdas, standardize = false; kwds...)
    LassoPathFits(fits, μ, σ)
end
ScikitLearn.predict(model::LassoPathFits, X) =
    StatsBase.predict(model.result, zscorecol(X, model.μ, model.σ), select = AllSeg())

function buildmodel(params, classifier, seed)
    model = if classifier == :svm_radial
        SVC(;params...)
    elseif classifier == :svm_linear
        SVC(
            kernel = "linear",
            random_state = stablehash(params, seed) & typemax(UInt32);
            params...
        )
    elseif classifier == :gradient_boosting
        GradientBoostingClassifier(
            loss             = "deviance",
            random_state     = stablehash(params, seed) & typemax(UInt32),
            n_iter_no_change = 10,
            max_features     = "auto";
            params...
        )
    elseif classifier == :logistic_l1
        @assert s(params) == __classifierparams__[:logistic_l1]
        LassoClassifier(values(params)...)
    else
        error("Unknown classifier $classifier.")
    end
end

function formulafn(data, y, X, Coding = DummyCoding)
    if @_ any(eltype(_) >: Missing, eachcol(view(data,:, X)))
        @warn "Some data columns have a missing type which will result in dummy coding. "*
            "This may not be what you intend."
    end

    # model formula (starts empty)
    formula = term(0)

    # include all columns `X` as features of the classification
    for col in propertynames(view(data, :, X))
        formula += term(col)
    end
    # include `y` as the dependent variable (the class to be learned)
    levels = CategoricalArrays.levels(data[:, y])
    yterm = StatsModels.CategoricalTerm(y, StatsModels.ContrastsMatrix(Coding(), levels))
    formula = yterm ~ formula
    f = apply_schema(formula, schema(formula, data))

    subdata ->  modelcols(f, subdata), levels
end

function runclassifier(model; data, y, X, seed = nothing, kwds...)
    getxy, levels = formulafn(data, y, X)

    _y, _X = getxy(data)
    coefs = ScikitLearn.fit!(model, _X, vec(_y); kwds...)

    level = ScikitLearn.predict(coefs, _X)
    _labels = levels[round.(Int, level).+1]

    result = copy(data)
    result[!,:label] = convert(Array{Union{String, Missing}}, _labels)
    result[!,:correct] = convert(Array{Union{Bool, Missing}}, _labels .== data[:, y])

    coefs, coefvals(coefs), result
end
coefvals(coefs::LassoClassifierFit) = StatsBase.coef(coefs.result)
function coefvals(coefs::PyObject)
    # TODO!!!
    coefs
end

seedmodel(model, seed) = Random.seed!(seed)
seedmodel(model::PyObject, seed) = numpy.random.seed(typemax(UInt32) & seed)

function paramvals(model::LassoPathClassifiers, fit::LassoPathFits, col, coefnames)
    @assert isempty(coefnames) "Model coefficient report is not supported"
    model.lambdas[col], sum(!iszero, @view(fit.result.coefs[:, col]))
end
function paramnames(model::LassoPathClassifiers, fit, coefnames)
    @assert isempty(coefnames) "Model coefficient report is not supported"
    :λ, :nzcoef
end
function paramvals(model::LassoPathClassifiers, fit::Nothing, coefnames)
    missing, missing
end

paramnames(model::LassoClassifier, fit, coefnames) = coefnames
function paramvals(model::LassoClassifier, fit, coefnames)
    isempty(coefnames) ? () : coef(fit.result)
end

function testclassifier(model; data, y, X, crossval, n_folds = 10,
    seed  = nothing, weight = nothing, on_model_exception = :debug,
    include_model_coefs = false, ycoding = DummyCoding, kwds...)
    @assert on_model_exception ∈ [:debug, :print, :throw]

    if !isnothing(seed); seedmodel(model, seed); end

    getxy, levels = formulafn(data, y, X, ycoding)

    # the results of classification (starts empty)
    result = Empty(DataFrame)

    # cross validation loop
    ids = shuffle!(stableRNG(seed, :testclassifier), unique(data[:, crossval]))
    _folds = folds(n_folds, unique(data[:, crossval]), on_all_empty_test = :nothing,
        rng = stableRNG(seed))
    for (i, (trainids, testids)) in enumerate(_folds)
        train = @_ filter(_[crossval] in trainids, data)
        test = @_ filter(_[crossval] in testids, data)

        # check for at least 2 classes
        function predictlabels()
            if length(unique(train[:, y])) < 2
                @warn "Degenerate classification (1 class), bypassing training" maxlog = 1
                return fill(missing, size(test, 1)), nothing
            end

            _y, _X = getxy(train)
            if eltype(_y) >: Missing && any(ismissing, _y)
                error("The `y` variable ($y) has missing values")
            end
            # if size(_y, 2) > 1
            #     error("The value for `y` ($y) should only reference a single, two-category column.")
            # end

            weigths_kwds = isnothing(weight) ? kwds : (wts = float(train[:, weight]), kwds...)

            local coefs_for_code
            try
                coefs_for_code = map(1:size(_y, 2)) do col
                    vals = sort!(unique(_y[:, col]))
                    yidx = indexin(_y[:, col], vals)
                    yt = [0.0, 1.0][yidx]
                    (coefs = ScikitLearn.fit!(model, _X, yt; weigths_kwds...), vals = vals)
                end
            catch e
                if on_model_exception == :debug
                    @info "Model fitting threw an error: opening debug to troubleshoot..."
                    @infiltrate
                    rethrow(e)
                elseif on_model_exception == :print
                    buffer = IOBuffer()
                    for (exc, bt) in Base.catch_stack()
                        showerror(buffer, exc, bt)
                        println(buffer)
                    end
                    @error "Exception while fitting model: $(String(take!(buffer)))"
                    return fill(missing, size(test, 1)), nothing
                elseif on_model_exception == :throw
                    rethrow(e)
                end
            end

            # test the model
            _y, _X = getxy(test)
            predicted = if length(coefs_for_code) > 1
                mapreduce((x, y) -> cat(x, y, dims = 3), coefs_for_code) do cfs
                    cfs.vals[1 .+ round.(Int, ScikitLearn.predict(cfs.coefs, _X))]
                end
            else
                cfs = first(coefs_for_code)
                p = cfs.vals[1 .+ round.(Int, ScikitLearn.predict(cfs.coefs, _X))]
                reshape(p, :, size(p, 2), 1)
            end
            C = StatsModels.ContrastsMatrix(ycoding(), levels)
            indices = [argmin(collect(eachcol(abs.(predicted[I,:] .- C.matrix'))))
                for I in CartesianIndices(size(predicted)[1:2])]

            if any(isnothing, indices)
                nothing_index = findfirst(isnothing, indices)
                Is = CartesianIndices(size(predicted)[1:2])
                row = predicted[Is[nothing_index],:]
                @error "Could not match $row to a contrast pattern."
            end

            return levels[indices], map(x -> x.coefs, coefs_for_code)
        end
        _labels, coefs_for_code = predictlabels()

        # add to the results
        keepvars = propertynames(view(data, :, Not(X)))
        label    = convert(Array{Union{String, Missing}}, _labels)
        correct  = convert(Array{Union{Bool, Missing}},   _labels .== test[:, y])

        coefnames = include_model_coefs ? pushfirst!(propertynames(data[:,X]),:C) : []
        if size(_labels,2) > 1
            for (mcol, coefs) in enumerate(coefs_for_code)
                for col in 1:size(_labels,2)
                    result = append!!(result, DataFrame(
                        modelcol   =  mcol,
                        label      =  @view(label[:,col]),
                        correct    =  @view(correct[:,col]),
                        label_fold =  i;
                        (keepvars .=> eachcol(test[:, keepvars]))...,
                        (paramnames(model, coefs, coefnames) .=>
                            paramvals(model, coefs, col, coefnames))...,
                    ))
                end
            end
        else
            for (mcol, coefs) in enumerate(coefs_for_code)
                result = append!!(result, DataFrame(
                    modelcol    =  mcol,
                    label       =  vec(label),
                    correct     =  vec(correct),
                    label_fold  =  i;
                    (keepvars  .=> eachcol(test[:, keepvars]))...,
                    (paramnames(model, coefs, coefnames) .=>
                        paramvals(model, coefs, coefnames))...,
                ))
            end
        end
    end

    result
end
