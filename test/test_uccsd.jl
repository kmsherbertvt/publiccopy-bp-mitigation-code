using Test
using AdaptBarren
using IterTools

@testset "Test that terms in ops commute, doubles" begin
    for n=4:6
        for (a,b,i,j)=product(1:n,1:n,1:n,1:n)
            if !(b>a>j>i) continue end
                @test subcommutes(cluster_doub_op(a, b, i, j, n))
        end
    end
end

@testset "Test that terms in ops commute, singles" begin
    for n=4:6
        for (p,q)=product(1:n,1:n)
            if p<=q continue end
            op = cluster_sing_op(p, q, n)
            @test subcommutes(op)
        end
    end
end