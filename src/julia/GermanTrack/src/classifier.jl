export testclassifier

function testclassifier(model;data, y, X, crossval, n_folds = 10, seed = nothing, kwds...)
    if !isnothing(seed)
        numpy.random.seed(typemax(UInt32) & seed)
    end

    # model formula (starts empty)
    formula = term(0)

    # include all columns `X` as features of the classification
    for col in propertynames(view(data, :, X))
        formula += term(col)
    end
    # include `y` as the dependent variable (the class to be learned)
    formula = term(y) ~ formula

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
            # setup the model and fit it
            f = apply_schema(formula, schema(formula, data))
            _y, _X = modelcols(f, train)
            local coefs
            try
                coefs = ScikitLearn.fit!(model, _X, vec(_y);kwds...)
            catch e
                @infiltrate
            end

            # test the model
            _y, _X = modelcols(f, test)
            level = ScikitLearn.predict(coefs, _X)
            f.lhs.contrasts.levels[round.(Int, level).+1]
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
