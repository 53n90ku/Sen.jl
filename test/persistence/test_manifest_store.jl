using Test
using Sen

@testset "manifest store" begin
    manifest=create_database_manifest(128, :cosine, 500; nlists = 10)

    @test manifest.format_version==1
    @test manifest.dim==128
    @test manifest.metric==:cosine
    @test manifest.count==500
    @test manifest.nlists==10

    mktempdir() do path
        manifest_path=save_manifest(path, manifest)

        @test isfile(manifest_path)

        loaded=load_manifest(path)

        @test loaded.format_version==manifest.format_version
        @test loaded.dim==manifest.dim
        @test loaded.metric==manifest.metric
        @test loaded.count==manifest.count
        @test loaded.nlists==manifest.nlists
    end

    @test_throws ArgumentError create_database_manifest(0, :cosine, 1)
    @test_throws ArgumentError create_database_manifest(2, :invalid, 1)
    @test_throws ArgumentError create_database_manifest(2, :cosine, -1)
    version_two=create_database_manifest(
        2,
        :cosine,
        4;
        format_version = 2,
        nlists = 2,
        revision = 7,
        index_revision = 7,
        iterations = 8,
        seed = 9,
        restarts = 2,
        training_count = 4,
    )

    @test version_two.format_version==2
    @test version_two.revision==7
    @test version_two.index_revision==7
    @test version_two.build_config==IndexBuildConfig(2, 8, 9, 2, 4)

    mktempdir() do path
        save_manifest(path, version_two)
        loaded=load_manifest(path)

        @test loaded.format_version==2
        @test loaded.revision==7
        @test loaded.index_revision==7
        @test loaded.build_config==version_two.build_config
    end

    version_three=create_database_manifest(
        2,
        :cosine,
        4;
        format_version = 3,
        nlists = 2,
        revision = 8,
        index_revision = 7,
        iterations = 8,
        seed = 9,
        restarts = 2,
        training_count = 4,
    )
    @test version_three.format_version==3
    @test version_three.index_revision==7

    mktempdir() do path
        save_manifest(path, version_three)
        @test load_manifest(path)==version_three
    end

    @test_throws ArgumentError create_database_manifest(2, :cosine, 1; format_version = 4)
    @test_throws ArgumentError create_database_manifest(
        2,
        :cosine,
        2;
        format_version = 2,
        nlists = 1,
        revision = 1,
        index_revision = 2,
    )
    @test_throws ArgumentError create_database_manifest(2, :cosine, 1; nlists = 2)

    mktempdir() do path
        @test_throws ArgumentError load_manifest(path)
    end
end
