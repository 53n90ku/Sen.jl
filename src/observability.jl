function database_index_bytes(index::FilterAwareIVFIndex, filter_index::BitsetIndex)
    return Base.summarysize((index, filter_index))
end

"""
    database_info(db)

Return a consistent immutable snapshot of database, index, durability and
background-maintenance state, including logical counts, index memory, revisions
and WAL checkpoint progress.
"""
function database_info(db::VectorDB)
    return with_database_read(db.database_lock) do
        ensure_database_open(db)
        counts=maintenance_counts_locked(db)
        index_count=db.index===nothing ? 0 : sum(length, db.index.ivf.lists)
        index_lists=db.index===nothing ? 0 : length(db.index.ivf.lists)
        built=has_usable_base(db)&&db.index_revision==db.revision
        dirty=db.revision>0&&(db.index_revision===nothing||db.index_revision!=db.revision)
        state=db.maintenance_state
        lock(state.lock)

        try
            return DatabaseInfo(
                db.path,
                db.dim,
                db.metric,
                db.live_count,
                counts.base_count,
                counts.delta_count,
                counts.delta_search_work,
                counts.delta_search_limit,
                counts.tombstone_count,
                counts.delta_ratio,
                counts.tombstone_ratio,
                db.revision,
                db.index_revision,
                index_count,
                index_lists,
                db.index_bytes,
                built,
                dirty,
                db.writer_lock!==nothing,
                db.wal_revision,
                db.wal_checkpoint_revision,
                db.maintenance_config.enabled,
                database_maintenance_due_locked(db),
                state.status,
                state.task!==nothing,
                state.attempts,
                state.last_completed_revision,
                state.last_duration_ms,
                state.last_error,
            )
        finally
            unlock(state.lock)
        end
    end
end
