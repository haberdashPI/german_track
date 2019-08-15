
using Test
include(joinpath(@__DIR__,"..","util","setup.jl"))


function simple_lags(x,lags)
    y = similar(x,size(x,1),size(x,2)*length(lags))
    for r in axes(x,1)
        for (l,lag) in enumerate(lags)
            for c in axes(x,2)
                r_ = r + lag
                if r_ <= 0
                    y[r,(l-1)*size(x,2)+c] = 0
                elseif r_ > size(x,1)
                    y[r,(l-1)*size(x,2)+c] = 0
                else
                    y[r,(l-1)*size(x,2)+c] = x[r_,c]
                end
            end
        end
    end

    y
end

@testset "Lag Mult" begin
    x = collect(float(reshape(1:12,4,3)))
    y = simple_lags(x,-1:1)
    @test y'*y ≈ lagouter(x,-1:1)

    y = simple_lags(x,-2:0)
    @test y'*y ≈ lagouter(x,-2:0)

    y = simple_lags(x,0:2)
    @test y'*y ≈ lagouter(x,0:2)

    x = rand(4,4)
    y = simple_lags(x,-1:1)
    @test isapprox(y'*y,lagouter(x,-1:1),atol=1e-8)

end
