using Test
using Sen

const ROOT=normpath(joinpath(@__DIR__,".."))
const WORKER=joinpath(@__DIR__,"crash_worker.jl")

function initialize_crash_database(path::AbstractString)
    db=create_db(path;dim=2,metric=:dot,checkpoint_operations=0,checkpoint_bytes=0,)
    insert!(db,[1.0,0.0],(source="baseline",);id="baseline",)
    save!(db)
    revision=db.revision
    close(db)
    return revision
end

function wait_for_crash_marker(process::Base.Process,marker::AbstractString,stderr_path::AbstractString;timeout_seconds::Float64=30.0,)
    deadline=time()+timeout_seconds

    while time()<deadline
        isfile(marker)&&return read(marker,String)

        if !Base.process_running(process)
            wait(process)
            details=isfile(stderr_path) ? read(stderr_path,String) : ""
            error("crash worker exited before reaching its boundary: $details")
        end

        sleep(0.01)
    end

    Base.process_running(process)&&kill(process,Base.SIGKILL)
    wait(process)
    details=isfile(stderr_path) ? read(stderr_path,String) : ""
    error("timed out waiting for crash boundary marker: $details")
end

function kill_at_database_boundary(path::AbstractString,stage::Symbol,mode::Symbol,record_id::AbstractString)
    marker=joinpath(dirname(path),"$(stage).marker")
    stdout_path=joinpath(dirname(path),"$(stage).stdout")
    stderr_path=joinpath(dirname(path),"$(stage).stderr")
    project="--project=$(ROOT)"
    command=`$(Base.julia_cmd()) --startup-file=no $project $WORKER $mode $path $record_id`
    command=addenv(command,
        Sen.DATABASE_CRASH_STAGE_ENV=>String(stage),
        Sen.DATABASE_CRASH_MARKER_ENV=>marker,
    )
    stdout_io=open(stdout_path,"w")
    stderr_io=open(stderr_path,"w")
    process=run(pipeline(ignorestatus(command),stdout=stdout_io,stderr=stderr_io);wait=false,)

    try
        marker_contents=wait_for_crash_marker(process,marker,stderr_path)
        occursin("stage=$(stage)",marker_contents)||error("crash worker reached the wrong boundary")
        kill(process,Base.SIGKILL)
        wait(process)
        success(process)&&error("crash worker unexpectedly exited successfully")
    finally
        Base.process_running(process)&&kill(process,Base.SIGKILL)
        Base.process_running(process)&&wait(process)
        close(stdout_io)
        close(stderr_io)
    end

    return nothing
end

function verify_recovered_database(path::AbstractString,record_id::AbstractString,expected_revision::UInt64;present::Bool,expected_source::AbstractString="crash-worker",)
    db=load_db(path)

    try
        @test db.revision==expected_revision

        if present
            @test get_record(db,record_id).metadata.source==expected_source
        else
            @test_throws KeyError get_record(db,record_id)
        end

        recovery_id="recovery-$(record_id)"
        insert!(db,[-1.0,0.0],(source="recovery",);id=recovery_id,)
        save!(db)
    finally
        close(db)
    end

    reopened=load_db(path)

    try
        @test get_record(reopened,"recovery-$(record_id)").metadata.source=="recovery"
    finally
        close(reopened)
    end
end

function exercise_storage_fault(stage::Symbol)
    mktempdir() do root
        path=joinpath(root,"database")
        initial_revision=initialize_crash_database(path)
        db=load_db(path)

        if stage===:wal_append
            wal_size=filesize(Sen.database_wal_path(path))
            Sen.inject_database_storage_fault!(stage)
            @test_throws Sen.InjectedDatabaseStorageError insert!(db,[0.0,1.0],(source="fault",);id="fault",)
            @test db.revision==initial_revision
            @test filesize(Sen.database_wal_path(path))==wal_size
            @test_throws KeyError get_record(db,"fault")
            close(db)
            verify_recovered_database(path,"fault",initial_revision;present=false,)
            return
        end

        insert!(db,[0.0,1.0],(source="fault",);id="fault",)
        expected_revision=db.revision
        old_generation=Sen.current_database_generation(path)
        old_generations=Sen.database_snapshot_generations(path)
        Sen.inject_database_storage_fault!(stage)
        @test_throws Sen.InjectedDatabaseStorageError save!(db)

        if stage===:snapshot_commit
            @test Sen.current_database_generation(path)==old_generation
            @test Sen.database_snapshot_generations(path)==old_generations
        elseif stage===:current_pointer
            @test Sen.current_database_generation(path)==old_generation
            @test length(Sen.database_snapshot_generations(path))==length(old_generations)+1
        elseif stage===:wal_checkpoint
            @test Sen.current_database_generation(path)!=old_generation
            @test length(Sen.read_database_wal(path).records)==1
        end

        close(db)
        verify_recovered_database(path,"fault",expected_revision;present=true,expected_source="fault",)
    end
end

@testset "process SIGKILL recovery matrix" begin
    Sys.isunix()||error("process crash recovery requires Unix signals")

    for(stage,mode,present) in (
        (:before_wal_append,:mutation,false),
        (:after_wal_sync,:mutation,true),
        (:after_snapshot_seal,:snapshot,true),
        (:after_snapshot_commit,:snapshot,true),
    )
        @testset "$stage" begin
            mktempdir() do root
                path=joinpath(root,"database")
                initial_revision=initialize_crash_database(path)
                record_id=String(stage)
                kill_at_database_boundary(path,stage,mode,record_id)
                expected_revision=present ? initial_revision+UInt64(1) : initial_revision
                verify_recovered_database(path,record_id,expected_revision;present=present,)
            end
        end
    end
end

@testset "ENOSPC-style storage failure matrix" begin
    for stage in Sen.DATABASE_STORAGE_FAULT_STAGES
        @testset "$stage" exercise_storage_fault(stage)
    end
end

@testset "damaged durable artifact recovery" begin
    @testset "truncated WAL tail" begin
        mktempdir() do root
            path=joinpath(root,"database")
            initialize_crash_database(path)
            db=load_db(path)
            insert!(db,[0.0,1.0],(source="complete",);id="complete",)
            expected_revision=db.revision
            wal_path=Sen.database_wal_path(path)
            complete_size=filesize(wal_path)
            open(wal_path,"a") do io
                write(io,UInt8[0x01,0x02,0x03])
            end
            close(db)

            recovered=load_db(path)
            @test recovered.revision==expected_revision
            @test get_record(recovered,"complete").metadata.source=="complete"
            @test filesize(wal_path)==complete_size
            @test !Sen.read_database_wal(path).incomplete_tail
            close(recovered)
        end
    end

    @testset "corrupt current snapshot" begin
        mktempdir() do root
            path=joinpath(root,"database")
            initialize_crash_database(path)
            db=load_db(path)
            first_snapshot=Sen.current_database_snapshot(path)
            insert!(db,[0.0,1.0],(source="second",);id="second",)
            save!(db)
            corrupted_snapshot=Sen.current_database_snapshot(path)
            open(joinpath(corrupted_snapshot,"vectors.bin"),"a") do io
                write(io,UInt8(0xff))
            end
            close(db)

            @test_throws ArgumentError load_db(path)
            recovered=recover_db(path)
            @test Sen.current_database_snapshot(path)==first_snapshot
            @test get_record(recovered,"baseline").metadata.source=="baseline"
            @test_throws KeyError get_record(recovered,"second")
            insert!(recovered,[-1.0,0.0],(source="after-recovery",);id="after-recovery",)
            save!(recovered)
            close(recovered)
        end
    end
end
