export testclassifier

function testclassifier(model;data,y,X,crossval,n_folds=10,seed=nothing,kwds...)
    if !isnothing(seed)
        numpy.random.seed(typemax(UInt32) & seed)
    end

    # model formula (starts empty)
    formula = term(0)

    # include all columns `X` as features of the classification
    for col in propertynames(view(data,:,X))
        formula += term(col)
    end
    # include `y` as the dependent variable (the class to be learned)
    formula = term(y) ~ formula

    # the results of classification (starts empty)
    result = Empty(DataFrame)

    # cross validation loop
    for (i, (trainids,testids)) in enumerate(folds(n_folds,unique(data[:,crossval])))
        # select train and test sets
        isempty(trainids) && continue
        train = @_ filter(_[crossval] in trainids,data)
        test = @_ filter(_[crossval] in testids,data)

        # check for at least 2 classes
        if length(unique(data[:,y])) < 2
            error("Something is wrong, there is only one class in this classification ",
                  "task.")
        end

        # setup the model and fit it
        f = apply_schema(formula, schema(formula, data))
        _y,_X = modelcols(f, train)
        coefs = ScikitLearn.fit!(model,_X,vec(_y);kwds...)

        # test the model
        _y,_X = modelcols(f, test)
        level = ScikitLearn.predict(coefs,_X)
        _labels = f.lhs.contrasts.levels[round.(Int,level).+1]

        # add to the results
        keepvars = propertynames(view(data,:,Not(X)))
        result = append!!(result, DataFrame(
            label = _labels,
            correct = _labels .== test[:,y];
            (keepvars .=> eachcol(test[:,keepvars]))...
        ))
    end

    result
end
