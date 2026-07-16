using Sen

length(ARGS)==3||error("usage: crash_worker.jl <mutation|snapshot> <database-path> <record-id>")
mode=Symbol(ARGS[1])
path=ARGS[2]
record_id=ARGS[3]
mode in (:mutation,:snapshot)||error("unsupported crash worker mode")

db=load_db(path)

try
    insert!(db,[0.0,1.0],(source="crash-worker",);id=record_id,)
    mode===:snapshot&&save!(db)
finally
    close(db)
end
