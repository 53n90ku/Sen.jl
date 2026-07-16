using Test
using Sen

@testset "public api" begin
    expected=(
        :VectorDB,
        :SearchResult,
        :PlannerConfig,
        :DatabaseInfo,
        :MaintenanceConfig,
        :FilterExpr,
        :Eq,
        :In,
        :Range,
        :And,
        :Or,
        :Not,
        :create_db,
        :insert!,
        :upsert!,
        :update!,
        :delete!,
        :get_record,
        :build!,
        :rebuild!,
        :compact!,
        :is_built,
        :is_dirty,
        :database_info,
        :configure_maintenance!,
        :maintenance_status,
        :wait_for_maintenance,
        :search,
        :plan_query,
        :save!,
        :load_db,
        :recover_db,
    )

    @test stable_api()===STABLE_API_V1
    @test stable_api()==expected
    @test length(unique(stable_api()))==length(stable_api())
    @test Set(names(Sen))==Set((:Sen,:STABLE_API_V1,:stable_api,expected...,))

    for name in stable_api()
        @test isdefined(Sen,name)
        @test Base.isexported(Sen,name)
    end
end
