"""
    @usepython

Like `using PyCall`, except that it properly sets up the virtual environment
defined in the `Pipfile.lock` for this project.
"""
macro usepython()
    ## NOTE: this has to be a macro, because `using` is a top-level statement,
    ## and attempting it here may fail, if the first approach to setting up the environment
    ## doesn't work
    quote
        using DrWatson
        let homedir = projectdir()
            try
                ENV["PYCALL_JL_RUNTIME_PYTHON"] =
                    strip(read(setenv(`pipenv --py`, dir=homedir), String))
                using PyCall
            catch
                ENV["PYTHON"] =
                    strip(read(setenv(`pipenv --py`, dir=homedir), String))
                import Pkg; Pkg.build("PyCall")
                using PyCall
            end
        end
    end
end
