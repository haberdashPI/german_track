
using Test
include(joinpath(@__DIR__,"..","util","setup.jl"))

function lagouter_(x,lags)
    y = withlags(x,lags)
    y'*y
end

@testset "Lag Mult" begin
    x = collect(float(reshape(1:12,4,3)))
    @test lagouter_(x,-1:1) ≈ lagouter(x,-1:1)

    y = simple_lags(x,-2:0)
    @test y'*y ≈ lagouter(x,-2:0)

    y = simple_lags(x,0:2)
    @test y'*y ≈ lagouter(x,0:2)

    x = rand(400,64)
    y = simple_lags(x,-16:0)
    @test isapprox(y'*y,lagouter(x,-16:0),atol=1e-8)

end
