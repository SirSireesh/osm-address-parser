// Parquet input driver on duckdb (ImportC, libduckdb.so).
//
// file → duckdb read_parquet → flat D arrays
//
// One in-memory connection per process, opened lazily, spatial
// extension loaded for the geometry reads. Row order stays the
// file order — read_parquet preserves it and stage-2 outputs
// depend on it.
//
// Text columns are requested by name; nulls become "" (the
// extractor already writes empty tag values as nulls, so both
// mean "absent" to callers).
module address_generator.parquet_input;

import std.array : join;
import std.conv : to;
import std.exception : enforce;
import std.string : replace, toStringz;

import duckdb;

struct Parquet
{
    string[] osmTypes;
    long[] osmIds;

    // requested text columns, "" = null
    string[][string] texts;
}

// row range [lo, hi) over file_row_number, so large inputs stream
// in chunks instead of one resident array set; the full range is
// the plain ordered scan
Parquet load(string path, string[] textCols,
    size_t lo = 0, size_t hi = size_t.max)
{
    auto q = Query("SELECT osm_type, osm_id" ~ textSelect(textCols)
        ~ " FROM read_parquet('" ~ esc(path)
        ~ "', file_row_number=true)"
        ~ " WHERE file_row_number >= " ~ lo.to!string
        ~ " AND file_row_number < " ~ hi.to!string
        ~ " ORDER BY file_row_number");

    Parquet pq;
    pq.osmTypes = q.strings(OSM_TYPE);
    pq.osmIds = q.int64s(OSM_ID);

    foreach (i, col; textCols)
        pq.texts[col] = q.strings(ID_COLS + i);

    return pq;
}

// row count of a parquet file, for chunk sizing
public size_t rowCount(string path)
{
    auto q = query("SELECT count(*) FROM read_parquet('"
        ~ esc(path) ~ "')");

    return cast(size_t) q.int64s(0)[0];
}

// all-string parquet (the vocab bucket files): requested
// columns, "" = null
string[][string] loadText(string path, string[] textCols)
{
    auto q = Query("SELECT " ~ join(textCols, ", ")
        ~ " FROM read_parquet('" ~ esc(path) ~ "')");

    string[][string] cols;

    foreach (i, col; textCols)
        cols[col] = q.strings(i);

    return cols;
}

// double columns (street/locality points): parquet null → NaN
double[][string] loadNums(string path, string[] doubleCols)
{
    auto q = Query("SELECT " ~ join(doubleCols, ", ")
        ~ " FROM read_parquet('" ~ esc(path) ~ "')");

    double[][string] cols;

    foreach (i, col; doubleCols)
        cols[col] = q.doubles(i);

    return cols;
}

// -------- duckdb engine core --------

private enum OSM_TYPE = 0;
private enum OSM_ID = 1;
private enum ID_COLS = 2;

// current query's rows stay alive until the scope ends;
// public so the query layer (address_generator.queries) can
// run its own SQL through the same connection
public struct Query
{
    private duckdb_result res;

    this(string sql)
    {
        // duckdb 2.0 sets both the failure state and the
        // result error, so one enforce carries both
        const state = duckdb_query(con(), toStringz(sql), &res);
        const err = duckdb_result_error(&res);

        enforce(state == DuckDBSuccess && err is null,
            "duckdb: " ~ (err is null ? "query failed" : fromZ(err))
                ~ "\n" ~ sql);
    }

    ~this()
    {
        duckdb_destroy_result(&res);
    }

    private size_t rows()
    {
        return cast(size_t) duckdb_row_count(&res);
    }

    string[] strings(size_t col)
    {
        auto vals = new string[](rows);

        foreach (size_t i; 0 .. rows())
        {
            if (duckdb_value_is_null(&res, col, i))
                continue;

            auto s = duckdb_value_string(&res, col, i);

            vals[i] = intern(s.data[0 .. s.size].idup);

            duckdb_free(cast(void*) s.data);
        }

        return vals;
    }

    long[] int64s(size_t col)
    {
        auto vals = new long[](rows);

        foreach (size_t i; 0 .. rows())
            if (!duckdb_value_is_null(&res, col, i))
                vals[i] = duckdb_value_int64(&res, col, i);

        return vals;
    }

    // D doubles default-init to NaN: nulls need no extra branch
    double[] doubles(size_t col)
    {
        auto vals = new double[](rows);

        foreach (size_t i; 0 .. rows())
            if (!duckdb_value_is_null(&res, col, i))
                vals[i] = duckdb_value_double(&res, col, i);

        return vals;
    }

}

// process-wide canonical string table: repeated vocab values
// (pool names, labels) share one GC allocation instead of a copy
// per query row — the D heap's string duplication comes from
// here, not from the distinct values themselves
private __gshared string[string] gCanon;

private string intern(string s)
{
    if (auto p = s in gCanon)
        return *p;

    gCanon[s] = s;

    return s;
}

// process-wide connection, opened on first use
__gshared duckdb_database gDb;
__gshared duckdb_connection gCon;

private duckdb_connection con()
{
    if (gCon !is null)
        return gCon;

    // open via config, all options are DB-initialization-only:
    // - allow_unsigned_extensions: the vendored libduckdb.so is
    //   a from-source build without the signing keys, official
    //   (signed) extension binaries cannot be verified — the
    //   extension itself is vendored in-repo, trust comes from
    //   that, not from the signature
    // - extension_directory: the vendored spatial extension,
    //   no runtime download, no cache writes into --rm
    //   containers
    // - memory_limit: DuckDB's default is 80% of RAM, but the D
    //   heap (loaded buckets, output rows) lives in the same
    //   process — on a 15GB box the default OOMs (2026-11-12).
    //   The buffer pool also keeps earlier queries' blocks, so
    //   cap the engine well below RAM to leave heap headroom
    //   and force earlier spilling
    duckdb_config cfg;

    enforce(duckdb_create_config(&cfg) == DuckDBSuccess,
        "duckdb_create_config failed");

    enforce(duckdb_set_config(cfg, "allow_unsigned_extensions",
        "true") == DuckDBSuccess, "duckdb_set_config failed");

    enforce(duckdb_set_config(cfg, "memory_limit", "4GB")
        == DuckDBSuccess, "duckdb_set_config failed");

    enforce(duckdb_set_config(cfg, "extension_directory",
        toStringz(extDir())) == DuckDBSuccess,
        "duckdb_set_config failed");

    char* openErr;

    enforce(duckdb_open_ext(null, &gDb, cfg, &openErr)
        == DuckDBSuccess && openErr is null,
        "duckdb_open failed"
            ~ (openErr is null ? "" : ": " ~ fromZ(openErr)));

    duckdb_destroy_config(&cfg);

    enforce(duckdb_connect(gDb, &gCon) == DuckDBSuccess,
        "duckdb_connect failed");

    run("LOAD spatial");

    return gCon;
}

// thirdparty/duckdb-extensions — three dirs up from bin/<exe>
// is the repo root; the extension cache layout beneath it
// carries the duckdb version and platform (v2.0.0/linux_amd64)
private string extDir()
{
    import std.file : thisExePath;
    import std.path : buildPath, dirName;
    import std.process : environment;

    // dub test exes live in a dub cache dir: static derivation
    // can't see the repo, point it via the env override
    if (auto dir = environment.get("DUCKDB_EXTENSIONS_DIR"))
        return dir;

    // four dirs up from <proj>/bin/<exe> is the repo root
    return buildPath(thisExePath.dirName.dirName.dirName.dirName,
        "thirdparty", "duckdb-extensions");
}

// close the shared connection, freeing all engine memory (temp
// tables, buffer pool) — the next query reopens it lazily.
// Stage 2's spatial phase leaves the pool pinned at the memory
// cap and starves the later per-chunk queries (OOM 2026-11-12)
public void resetDb()
{
    if (gCon !is null)
    {
        duckdb_disconnect(&gCon);
        gCon = null;
    }

    if (gDb !is null)
    {
        duckdb_close(&gDb);
        gDb = null;
    }
}

// query-layer entry points: run SQL through the shared
// connection; Query reads the result columns, exec discards them
public Query query(string sql)
{
    return Query(sql);
}

public void exec(string sql)
{
    run(sql);
}

// string literal into sql: '' doubles the quote
public string escSql(string s)
{
    return esc(s);
}

// statement without a result consumer (LOAD, the test
// fixtures below write through it)
private void run(string sql)
{
    duckdb_result res;

    const state = duckdb_query(con(), toStringz(sql), &res);
    const err = duckdb_result_error(&res);

    enforce(state == DuckDBSuccess && err is null,
        "duckdb: " ~ (err is null ? "query failed" : fromZ(err))
            ~ "\n" ~ sql);

    duckdb_destroy_result(&res);
}

// "..., col" — text columns tail the identity pair
private string textSelect(string[] cols)
{
    return cols.length ? ", " ~ join(cols, ", ") : "";
}

// path literal into sql: '' doubles the quote
private string esc(string path)
{
    return path.replace("'", "''");
}

private string fromZ(const(char)* c)
{
    import core.stdc.string : strlen;

    return c[0 .. strlen(c)].idup;
}

unittest
{
    // fixture: all pull shapes in one parquet file
    import std.math : isNaN;

    run(`COPY (
            SELECT 'R' AS osm_type, 17168010 AS osm_id,
                   'MG Road' AS name, 12.97 AS lat, 77.59 AS lon
            UNION ALL
            SELECT 'R', 42, 'Church Street', NULL, NULL
            UNION ALL
            SELECT 'N', 7, NULL, NULL, NULL
        ) TO '` ~ FIXTURE ~ `' (FORMAT parquet)`);

    auto pq = load(FIXTURE, ["name"]);
    assert(pq.osmTypes == ["R", "R", "N"]);
    assert(pq.osmIds == [17168010, 42, 7]);
    assert(pq.texts["name"] == ["MG Road", "Church Street", ""]);

    auto texts = loadText(FIXTURE, ["name"]);
    assert(texts["name"] == ["MG Road", "Church Street", ""]);

    auto nums = loadNums(FIXTURE, ["lat", "lon"]);
    assert(nums["lat"][0] == 12.97);
    assert(nums["lon"][1].isNaN);
    assert(nums["lat"][2].isNaN && nums["lon"][2].isNaN);

    // row ranges stream a file in file order
    assert(rowCount(FIXTURE) == 3);

    auto mid = load(FIXTURE, ["name"], 1, 3);
    assert(mid.osmIds == [42, 7]);
    assert(mid.texts["name"] == ["Church Street", ""]);

    // interning: the same value read twice shares one allocation
    auto again = loadText(FIXTURE, ["name"])["name"];
    assert(texts["name"][0].ptr == again[0].ptr);
    assert(texts["name"][1].ptr == again[1].ptr);
}

private enum string FIXTURE = "/tmp/parquet_input_test.parquet";
