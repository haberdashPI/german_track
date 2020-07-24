export runclassifier, testclassifier, buildmodel, classifierparams, classifier_param_names

const __classifierparams__ = (
    svm_radial        = (:C,:gamma),
    svm_linear        = (:C,),
    gradient_boosting = (:max_depth, :n_estimators, :learning_rate),
    logistic_l1       = (:C,),
)
classifier_param_names(classifier) = __classifierparams__[classifier]
function classifierparams(obj, classifier)
    (;(p => obj[p] for p in __classifierparams__[classifier])...)
end

function buildmodel(params, classifier, seed)
    model = if classifier == :svm_radial
        SVC(;params...)
    elseif classifier == :svm_linear
        SVC(
            kernel = "linear",
            random_state = hash((params, seed)) & typemax(UInt32);
            params...
        )
    elseif classifier == :gradient_boosting
        GradientBoostingClassifier(
            loss             = "deviance",
            random_state     = hash((params, seed)) & typemax(UInt32),
            n_iter_no_change = 10,
            max_features     = "auto";
            params...
        )
    elseif classifier == :logistic_l1
        LogisticRegression(
            penalty      = "l1",
            random_state = hash((params, seed)) & typemax(UInt32),
            solver       = "liblinear";
            params...
        )
    end
end

function formulafn(data, y, X)
    # model formula (starts empty)
    formula = term(0)

    # include all columns `X` as features of the classification
    for col in propertynames(view(data, :, X))
        formula += term(col)
    end
    # include `y` as the dependent variable (the class to be learned)
    formula = term(y) ~ formula
    f = apply_schema(formula, schema(formula, data))

    subdata -> modelcols(f, subdata), f.lhs.contrasts.levels
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

    model, result
end

function testclassifier(model;data, y, X, crossval, n_folds = 10, seed = nothing, kwds...)
    if !isnothing(seed)
        numpy.random.seed(typemax(UInt32) & seed)
    end

    getxy, levels = formulafn(data, y, X)

    # the results of classification (starts empty)
    result = Empty(DataFrame)

    # cross validation loop
    _folds = folds(n_folds, unique(data[:, crossval]), on_all_empty_test = :nothing)
    for (i, (trainids, testids)) in enumerate(_folds)
        train = @_ filter(_[crossval] in trainids, data)
        test = @_ filter(_[crossval] in testids, data)

        # check for at least 2 classes
        _labels = if length(unique(train[:, y])) < 2
            @warn "Degenerate classification (1 class), bypassing training" maxlog = 1
            fill(missing, size(test, 1))
        else
            _y, _X = getxy(train)
            local coefs
            try
                coefs = ScikitLearn.fit!(model, _X, vec(_y);kwds...)
            catch e
                @infiltrate
            end

            # test the model
            _y, _X = getxy(test)
            level = ScikitLearn.predict(coefs, _X)
            levels[round.(Int, level).+1]
        end

        # add to the results
        keepvars = propertynames(view(data, :, Not(X)))
        result = append!!(result, DataFrame(
            label = convert(Array{Union{String, Missing}}, _labels),
            correct = convert(Array{Union{Bool, Missing}}, _labels .== test[:, y]);
            (keepvars .=> eachcol(test[:, keepvars]))...
        ))
    end

    result
end
