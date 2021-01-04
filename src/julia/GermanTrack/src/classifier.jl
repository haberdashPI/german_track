export runclassifier, testclassifier, LassoPathClassifiers, LassoClassifier, ZScoring,
    NullSelect, traintest

import StatsModels: DummyCoding, FullDummyCoding
export DummyCoding, FullDummyCoding

struct ZScoringFit{F,B,T}
    fit::F
    groupings::B
    μ::Vector{T}
    σ::Vector{T}
end
struct ZScoring{M,G}
    parent::M
    groupings::G
end
ZScoring(parent) = ZScoring(parent, :)
mapgroupings(X, fn, groupings) = map(bins -> fn(view(X, :, bins)), groupings)
mapgroupings(X, fn, ::Colon) = fn(X, dims = 1)
enumgroupings(X, groupings) = enumerate(groupings)
enumgroupings(X, ::Colon) = enumerate(axes(X, 2))

Lasso.MinCVmse(m::ZScoringFit, args...) = MinCVmse(m.fit, args...)
Lasso.MinCV1se(m::ZScoringFit, args...) = MinCV1se(m.fit, args...)

function StatsBase.fit(model::ZScoring, X, y, args...; kwds...)
    μ = mapgroupings(X, mean, model.groupings)
    for (i, bins) in enumgroupings(X, model.groupings)
        X[:, bins] .-= μ[i]
    end

    σ = mapgroupings(X, std, model.groupings)
    for (i, bins) in enumgroupings(X, model.groupings)
        X[:, bins] ./= σ[i]
    end

    result = fit(model.parent, X, y; kwds...)

    ZScoringFit(result, model.groupings, μ, σ)
end

function StatsBase.predict(fit::ZScoringFit, X, args...; kwds...)
    for (i, bins) in enumgroupings(X, fit.groupings)
        X[:, bins] .-= fit.μ[i]
    end

    for (i, bins) in enumgroupings(X, fit.groupings)
        X[:, bins] ./= fit.σ[i]
    end

    predict(fit.fit, X, args...; kwds...)
end

StatsBase.coef(fit::ZScoringFit, args...; kwds...) = StatsBase.coef(fit.fit, args...; kwds...)

struct NullSelect <: SegSelect
end
Lasso.segselect(path::Lasso.RegularizationPath, select::NullSelect) = 1

function traintest(df, fold; y, X = r"channel", selector = m -> MinAICc(), weight = nothing)
    train = filter(x -> x.fold != fold, df)
    test  = filter(x -> x.fold == fold, df)

    vals = unique(df[:, y])
    @assert vals |> length == 2

    initmodel(selector) = ZScoring(LassoPath, [(0:29) .+ i for i in 1:30:150])
    initmodel(selector::Number)  = ZScoring(LassoModel, [(0:29) .+ i for i in 1:30:150])
    initmodelkwds(selector) = (;)
    initmodelkwds(selector::Number) = (;λ = [selector])

    model = fit(initmodel(selector),
        Array(train[:,X]), train[:, y] .== first(vals), Bernoulli(), standardize = false,
        maxncoef = size(view(train,:,X), 2),
        wts = isnothing(weight) ? ones(size(train, 1)) : float(train[:, weight]);
        initmodelkwds(selector)...
    )

    predictmodel(model, x, selector) = predict(model, x, select = selector(model))
    predictmodel(model, x, selector::Number) = predict(model, x)
    ŷ = predictmodel(model, Array(test[:,X]), selector)

    test.predict = vals[ifelse.(ŷ .> 0.5, 1, 2)]
    test.correct = test.predict .== test[:, y]

    test, model
end

"""
    LassoClassifier(λ)

Define a single, logistic regression classifier with L1 regularization λ. Use this once
you've used cross-validation to pick λ. Handles z-scoring of all input variables
using the data from a call to `fit`.
"""
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

function StatsBase.fit(model::LassoClassifier, X, y; kwds...)
    μ = mean(X, dims = 1)
    σ = std(X, dims = 1)

    fit = StatsBase.fit(LassoModel, zscorecol(X, μ, σ), y,
        Bernoulli(), λ = [model.lambda], standardize = false; kwds...)
    LassoClassifierFit(fit, μ, σ)
end
StatsBase.predict(model::LassoClassifierFit, X) =
    StatsBase.predict(model.result, zscorecol(X, model.μ, model.σ))

"""
    LassoPathClassifiers(λs)

Define a series of logistic regression classifiers using the given values for λ. Use this
to pick a λ via cross-validation. Handles z-scoring of all input variables using
the data from a call to `fit`.
"""
struct LassoPathClassifiers
    lambdas::Vector{Float64}
end
struct LassoPathFits{T}
    result::T
    μ::Array{Float64,2}
    σ::Array{Float64,2}
end
function StatsBase.fit(model::LassoPathClassifiers, X, y; kwds...)
    μ = mean(X, dims = 1)
    σ = std(X, dims = 1)

    fits = StatsBase.fit(LassoPath, zscorecol(X, μ, σ), y,
        Bernoulli(), λ = model.lambdas, standardize = false; kwds...)
    LassoPathFits(fits, μ, σ)
end
StatsBase.predict(model::LassoPathFits, X) =
    StatsBase.predict(model.result, zscorecol(X, model.μ, model.σ), select = AllSeg())

"""
    formulafn(data, y, X, Coding = DummyCoding)

Helper function to handle organizating data into a matrix. Returns a function getxy
which can extract the y and X matrices for a given subset of data and a list of all
levels of y.

You can customize how y is coded, as per [`StatModels`](https://github.com/JuliaStats/StatsModels.jl).
"""
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

"""
    runclassifier(model; data, y, X, seed, fit_kwds...)

Fit the given model using data from `data` for column `y` and columns `X`. Use the given
`seed` to ensure reproducability. `fit_kwds` will be model specific parameters.
"""
function runclassifier(model; data, y, X, seed = nothing, kwds...)
    getxy, levels = formulafn(data, y, X)

    _y, _X = getxy(data)
    coefs = fit(model, _X, vec(_y); kwds...)

    level = predict(coefs, _X)
    _labels = levels[round.(Int, level).+1]

    result = copy(data)
    result[!,:label] = convert(Array{Union{String, Missing}}, _labels)
    result[!,:correct] = convert(Array{Union{Bool, Missing}}, _labels .== data[:, y])

    coefs, coefvals(coefs), result
end
coefvals(coefs::LassoClassifierFit) = StatsBase.coef(coefs.result)

seedmodel(model, seed) = Random.seed!(seed)

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

"""
    runclassifier(model; data, y, X, seed, crossval, n_folds = 10, weight,
        on_model_exception = :debug, include_model_coefs = false,
        fit_kwds...)

Fit the given model using data from `data` for column `y` and columns `X`, then test it
using cross validation with `n_folds` folds. Use the given `seed` to ensure reproducability. `fit_kwds` will be
model specific parameters.

Results are returned as a data frame, with appropriately named columns.

## Additional keyword args
- `weight`: a column to weight individual observations in `data`
- `on_model_exception`: what to do when an exception occurs in the mode; options include:
    - `:debug`: runs @infiltrate at the site of the error allowing user
        to examine data (refer to `runclassifier` source for other relevant variables).
        **NOT THREAD SAFE**
    - `:print`: display the error, but use missing values for the particular fold
        where the error occured.
    - `:error` (default): throw an error
- `on_missing_case`
    - `:error` (default): throw an error if there is only one label value for `y`
    - `:missing`: return missing data if there is only one label value for `y`
    - `:debug`: call @infiltrate; **NOT THREAD SAFE**
- `include_model_coefs`: Return the coefficients as columns in the resulting data, useful
 for display and interpretation of the model.
"""
function testclassifier(model; data, y, X, crossval, n_folds = 10,
    seed  = nothing, weight = nothing, on_model_exception = :error,
    on_missing_case = :error, include_model_coefs = false, ycoding = DummyCoding, kwds...)
    @assert on_model_exception ∈ [:debug, :print, :error]

    if !isnothing(seed); seedmodel(model, seed); end

    if data[:,y] |> unique |> length == 1
        if on_missing_case == :missing
            return Empty(DataFrame)
        elseif on_missing_case == :debug
            @infiltrate
            rethrow(e)
        else
            error("Degenerate case (1 class present): ", data[1, Not(X)])
        end
    end

    local getxy, levels
    try
        getxy, levels = formulafn(data, y, X, ycoding)
    catch e
        if on_model_exception == :debug
            @info "Model setup threw an error: opening debug..."
            @infiltrate
            rethrow(e)
        else
            rethrow(e)
        end
    end

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
                    (coefs = fit(model, _X, yt; weigths_kwds...), vals = vals)
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
                elseif on_model_exception == :error
                    rethrow(e)
                end
            end

            # test the model
            _y, _X = getxy(test)
            predicted = if length(coefs_for_code) > 1
                mapreduce((x, y) -> cat(x, y, dims = 3), coefs_for_code) do cfs
                    cfs.vals[1 .+ round.(Int, predict(cfs.coefs, _X))]
                end
            else
                cfs = first(coefs_for_code)
                p = cfs.vals[1 .+ round.(Int, predict(cfs.coefs, _X))]
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
