// Set-based bucket queries on duckdb: the data-side operations
// that ran as D loops over loaded arrays — vocab anti-joins,
// nearest-road imputation, nearest-locality pools, (locality,
// street) pair enumeration, point-in-polygon. SQL mechanics live
// here and in parquet_input; callers get domain-typed results.
// Row order stays the parquet file order (read_parquet preserves
// it, file_row_number pins it) and stage-2 outputs depend on it,
// so every query either keeps scan order or orders explicitly
// with the D tie-breaks: nearest wins on distance, the earlier
// file row wins on ties
module address_generator.queries;

import std.array : join;
import std.conv : to;
import std.exception : enforce;
import std.format : format;
import std.math : isNaN;

import address_generator.coords : DEG_TO_M;
import address_generator.parquet_input : Query, query, exec, escSql;

// one (locality, street) coverage pair: the nearest door's
// city/postcode ride along
public struct PairRow
{
    string locality;
    string label;
    string street;
    string city;
    string postcode;
}

// one area containing a query point
public struct AreaHit
{
    string osmType;
    long osmId;
    string name;
    string label;
    string adminLevel;
    double m2;
}

// anti-join subselect: lowercased names of one vocab file;
// coalesce keeps null names comparable as "" (the D filter's
// null->"" mapping keeps them, "" is in no vocab)
private string antiJoin(string path)
{
    return "SELECT coalesce(lower(name), '') FROM read_parquet('"
        ~ escSql(path) ~ "')";
}

// units.parquet names minus the lowercased locality/city vocab
// (the addr:unit bulk-import cross-filter, buckets.d)
public string[] unitNames(string dir)
{
    auto q = query(
        "SELECT coalesce(u.name, '') FROM read_parquet('"
        ~ escSql(dir ~ "/units.parquet") ~ "') u WHERE coalesce(lower(u.name), '') NOT IN ("
        ~ antiJoin(dir ~ "/localities.parquet") ~ ") AND coalesce(lower(u.name), '') NOT IN ("
        ~ antiJoin(dir ~ "/cities.parquet") ~ ")");

    return q.strings(0);
}

// the street vocab: distinct road names from streets.parquet
// sidecar minus the same anti-join; the locality-shape predicate
// (gates.isLocalityStreet) stays in D, it is a pure string test,
// not a data join
public string[] roadNames(string dir)
{
    auto q = query(
        "SELECT DISTINCT s.name FROM read_parquet('"
        ~ escSql(dir ~ "/streets.parquet") ~ "') s WHERE coalesce(lower(s.name), '') NOT IN ("
        ~ antiJoin(dir ~ "/localities.parquet") ~ ") AND coalesce(lower(s.name), '') NOT IN ("
        ~ antiJoin(dir ~ "/cities.parquet") ~ ") ORDER BY s.name");

    return q.strings(0);
}

// ---- nearest-road queries over streets.parquet ----

// streets.parquet (one line per named road way) as a temp table with an
// RTREE, shared by the imputation and the spatial anchors
private void ensureStreets(string dir)
{
    exec("CREATE OR REPLACE TEMP TABLE rl AS SELECT file_row_number AS seq,"
        ~ " name, geometry AS g FROM read_parquet('"
        ~ escSql(dir ~ "/streets.parquet")
        ~ "', file_row_number=true)");

    exec("CREATE INDEX rl_g_idx ON rl USING RTREE (g)");
}

// the imputation as SQL: nearest-within-radius per address row.
// maxDeg is the radius in degrees; ties keep the earlier
// streets row
private string sqlImpute(string dir, double maxDeg)
{
    ensureStreets(dir);

    return "SELECT a.file_row_number AS seq, coalesce(r.name, '') AS name FROM"
        ~ " read_parquet('" ~ escSql(dir ~ "/addresses.parquet")
        ~ "', file_row_number=true) a"
        ~ " LEFT JOIN rl r ON ST_DWithin(ST_Point(a.lon, a.lat), r.g, "
        ~ format("%.17g", maxDeg) ~ ")"
        ~ " QUALIFY row_number() OVER (PARTITION BY a.file_row_number"
        ~ " ORDER BY ST_Distance(ST_Point(a.lon, a.lat), r.g), r.seq) = 1";
}

// one imputed street per addresses.parquet row, "" without coords
// or without a road point in range (wholesale imputation,
// NOTES 2026-10-26); rows is the addresses row count
public string[] imputeStreets(string dir, size_t rows, double maxDeg)
{
    auto q = query(sqlImpute(dir, maxDeg));

    auto seqs = q.int64s(0);
    auto names = q.strings(1);

    auto out_ = new string[](rows);

    foreach (i; 0 .. seqs.length)
    {
        enforce(seqs[i] >= 0 && cast(size_t) seqs[i] < rows,
            "imputation row out of range");
        out_[seqs[i]] = names[i];
    }

    return out_;
}

// per address row: the nearest real way point of the row's
// imputed street within rangeDeg of the row itself (the
// injectLocal anchor). Returns ["lat"]/["lon"] arrays of length
// rows, NaN where the row has no coords or no such point;
// ties keep the earlier streets row
public double[][string] roadAnchorByRow(string dir, size_t rows,
    double imputeDeg, double rangeDeg)
{
    auto q = query(
        "WITH imp AS (" ~ sqlImpute(dir, imputeDeg) ~ "),"
        ~ " a AS (SELECT file_row_number AS seq, lat, lon"
        ~ " FROM read_parquet('" ~ escSql(dir ~ "/addresses.parquet")
        ~ "', file_row_number=true)"
        ~ " WHERE lat IS NOT NULL AND lon IS NOT NULL"
        ~ " AND NOT isnan(lat) AND NOT isnan(lon)),"
        ~ " j AS (SELECT a.seq,"
        ~ " ST_Y(ST_ClosestPoint(p.g, ST_Point(a.lon, a.lat))) AS lat,"
        ~ " ST_X(ST_ClosestPoint(p.g, ST_Point(a.lon, a.lat))) AS lon,"
        ~ " ST_Distance(ST_Point(a.lon, a.lat), p.g) AS d,"
        ~ " p.seq AS rseq FROM a JOIN imp i ON i.seq = a.seq"
        ~ " JOIN rl p ON p.name = i.name AND ST_DWithin("
        ~ " ST_Point(a.lon, a.lat), p.g, "
        ~ format("%.17g", rangeDeg) ~ "))"
        ~ " SELECT seq, lat, lon FROM j"
        ~ " QUALIFY row_number() OVER (PARTITION BY seq"
        ~ " ORDER BY d, rseq) = 1");

    auto lat = new double[](rows);
    auto lon = new double[](rows);

    foreach (i; 0 .. rows)
    {
        lat[i] = double.nan;
        lon[i] = double.nan;
    }

    auto seqs = q.int64s(0);
    auto lats = q.doubles(1);
    auto lons = q.doubles(2);

    foreach (i; 0 .. seqs.length)
    {
        enforce(seqs[i] >= 0 && cast(size_t) seqs[i] < rows,
            "anchor row out of range");
        lat[seqs[i]] = lats[i];
        lon[seqs[i]] = lons[i];
    }

    double[][string] out_;
    out_["lat"] = lat;
    out_["lon"] = lon;

    return out_;
}

// per locality: up to k distinct road names with a real way
// point within rangeDeg of the locality point. No fallback for
// an empty range — a locality with no road in reach gets no
// pool entry and synth rows draw from the full vocab instead
// (dataset.synthComponents); a k-nearest-overall fallback is
// 56k misses x 146k streets and spills past disk (2026-11-12).
// Ties keep the earlier streets row. Keyed by locality name
public string[][string] roadPoolByLocality(string dir,
    double rangeDeg, size_t k)
{
    ensureStreets(dir);

    auto q = query(
        "WITH locs AS (SELECT name, ST_Point(lon, lat) AS g"
        ~ " FROM read_parquet('" ~ escSql(dir ~ "/localities.parquet") ~ "')"
        ~ " WHERE lat IS NOT NULL AND lon IS NOT NULL"
        ~ " AND NOT isnan(lat) AND NOT isnan(lon)),"
        ~ " inr AS (SELECT l.name AS lname, p.name AS rname,"
        ~ " min(ST_Distance(l.g, p.g)) AS d, min(p.seq) AS rseq"
        ~ " FROM locs l JOIN rl p ON ST_DWithin(l.g, p.g, "
        ~ format("%.17g", rangeDeg) ~ ") GROUP BY l.name, p.name),"
        ~ " poolA AS (SELECT lname, rname, d, rseq FROM ("
        ~ " SELECT lname, rname, d, rseq, row_number() OVER"
        ~ " (PARTITION BY lname ORDER BY d, rseq) AS rn FROM inr)"
        ~ " WHERE rn <= " ~ k.to!string ~ ")"
        ~ " SELECT lname, rname FROM poolA ORDER BY lname, d, rseq");

    string[][string] out_;

    auto lnames = q.strings(0);
    auto rnames = q.strings(1);

    foreach (i; 0 .. lnames.length)
        out_[lnames[i]] ~= rnames[i];

    return out_;
}

// one locality candidate in a per-row pool
public struct LocRef
{
    string name;
    string label;
}

// ---- nearest-locality pools over localities.parquet ----

// per anchor point: up to k localities within rangeDeg of the
// anchor; an empty range yields an empty pool (the row gets no
// injected locality). Ties keep the earlier localities row. A
// NaN anchor yields the full vocab, the uniform pool. firstRow is
// the addresses row index of lats[0]; the result aligns with the
// input arrays. Anchors go in as a temp table, chunked by the
// caller so rows × k pools never materialize at once
public LocRef[][] localityPoolByRow(string dir, size_t firstRow,
    const(double)[] lats, const(double)[] lons, double rangeDeg,
    size_t k)
{
    // locs + its RTREE are chunk-invariant: build them once per
    // session, not per chunk (a per-chunk rebuild re-read the whole
    // localities parquet and re-indexed it ~nRows/POOL_CHUNK times)
    exec("CREATE TEMP TABLE IF NOT EXISTS locs AS SELECT file_row_number AS lseq,"
        ~ " name, label, ST_Point(lon, lat) AS g"
        ~ " FROM read_parquet('" ~ escSql(dir ~ "/localities.parquet")
        ~ "', file_row_number=true)");

    exec("CREATE INDEX IF NOT EXISTS locs_g_idx ON locs USING RTREE (g)");

    exec("CREATE OR REPLACE TEMP TABLE anchors(seq BIGINT, pt GEOMETRY)");

    auto out_ = new LocRef[][](lats.length);

    if (lats.length == 0)
        return out_;

    string[] chunk;

    void flush()
    {
        if (chunk.length == 0)
            return;

        exec("INSERT INTO anchors VALUES " ~ chunk.join(","));

        chunk = null;
    }

    foreach (i; 0 .. lats.length)
    {
        auto pt = (!isNaN(lats[i]) && !isNaN(lons[i]))
            ? "ST_Point(" ~ format("%.17g", lons[i]) ~ ", "
                ~ format("%.17g", lats[i]) ~ ")"
            : "NULL";

        chunk ~= "(" ~ (firstRow + i).to!string ~ ", " ~ pt ~ ")";

        if (chunk.length == 2000)
            flush();
    }

    flush();

    // in-range pool; NaN anchors get the full vocab in file
    // order. No k-nearest fallback for an empty range — that is
    // chunk x 92k localities of CROSS JOIN and spills past disk
    auto q = query(
        "WITH anch AS (SELECT seq, pt FROM anchors WHERE pt IS NOT NULL),"
        ~ " inr AS (SELECT a.seq, l.name, l.label,"
        ~ " ST_Distance(a.pt, l.g) AS d, l.lseq"
        ~ " FROM anch a JOIN locs l ON ST_DWithin(a.pt, l.g, "
        ~ format("%.17g", rangeDeg) ~ ")),"
        ~ " poolA AS (SELECT seq, name, label, d, lseq FROM ("
        ~ " SELECT seq, name, label, d, lseq, row_number() OVER"
        ~ " (PARTITION BY seq ORDER BY d, lseq) AS rn FROM inr)"
        ~ " WHERE rn <= " ~ k.to!string ~ ")"
        ~ " SELECT seq, name, label, d, lseq FROM poolA"
        ~ " ORDER BY seq, d, lseq");

    // Query has a destructor: read it in place, never copy it
    // (a copy double-frees the duckdb result and segfaults)
    auto seqs = q.int64s(0);
    auto names = q.strings(1);
    auto labels = q.strings(2);

    foreach (i; 0 .. seqs.length)
    {
        enforce(seqs[i] >= cast(long) firstRow
            && cast(size_t)(seqs[i] - firstRow) < lats.length,
            "locality pool row out of range");

        out_[seqs[i] - firstRow] ~= LocRef(names[i], labels[i]);
    }

    // NaN anchors get the full vocab in file order — filled in D
    // from one vocab read, not a per-chunk CROSS JOIN (NULL anchors
    // are ~600/chunk, x 92k localities OOMs the engine)
    bool anyNull;

    foreach (i; 0 .. lats.length)
        if (isNaN(lats[i]) || isNaN(lons[i]))
            anyNull = true;

    if (anyNull)
    {
        // the full vocab is chunk-invariant: read it once per
        // process and share the array (pools are read-only
        // downstream); a per-chunk rebuild materialized the whole
        // locality vocab every POOL_CHUNK
        static LocRef[] full;
        static bool haveFull;

        if (!haveFull)
        {
            auto vq = query("SELECT name, label FROM locs ORDER BY lseq");
            auto vnames = vq.strings(0);
            auto vlabels = vq.strings(1);

            foreach (i; 0 .. vnames.length)
                full ~= LocRef(vnames[i], vlabels[i]);

            haveFull = true;
        }

        foreach (i; 0 .. lats.length)
            if (isNaN(lats[i]) || isNaN(lons[i]))
                out_[i] = full;
    }

    return out_;
}

// ---- (locality, street) coverage pairs ----

// allowedStreets becomes a temp table (the D-side street
// predicates — drop set, locality shape, ordinal-only — select
// the names before the query); chunked VALUES inserts
private void fillAllowed(bool[string] allowed)
{
    exec("CREATE OR REPLACE TEMP TABLE allowed(name VARCHAR)");

    string[] chunk;

    void flush()
    {
        if (chunk.length == 0)
            return;

        exec("INSERT INTO allowed VALUES "
            ~ chunk.join(",") ~ "");

        chunk = null;
    }

    foreach (name; allowed.byKey)
    {
        chunk ~= "('" ~ escSql(name) ~ "')";

        if (chunk.length == 2000)
            flush();
    }

    flush();
}

// every (locality, street-with-a-door-nearby) pair, one row each:
// nearest door per pair, maxPerStreet nearest localities per
// street, emission order locality-major. rangeDeg is the anchor
// radius in degrees; imputeDeg is the street-imputation radius
// the door's street was computed with
public PairRow[] pairList(string dir, bool[string] allowedStreets,
    double imputeDeg, double rangeDeg, size_t maxPerStreet)
{
    fillAllowed(allowedStreets);

    auto q = query(
        "WITH imp AS (" ~ sqlImpute(dir, imputeDeg) ~ "),"
        ~ " doors AS (SELECT a.file_row_number AS seq, i.name AS street,"
        ~ " a.lat, a.lon, coalesce(a.city, '') AS city,"
        ~ " coalesce(a.postcode, '') AS postcode"
        ~ " FROM read_parquet('" ~ escSql(dir ~ "/addresses.parquet")
        ~ "', file_row_number=true) a"
        ~ " JOIN imp i ON i.seq = a.file_row_number"
        ~ " JOIN allowed al ON al.name = i.name"
        ~ " WHERE a.lat IS NOT NULL AND a.lon IS NOT NULL"
        ~ " AND NOT isnan(a.lat) AND NOT isnan(a.lon)),"
        ~ " locs AS (SELECT name, label, ST_Point(lon, lat) AS g"
        ~ " FROM read_parquet('" ~ escSql(dir ~ "/localities.parquet") ~ "')"
        ~ " WHERE lat IS NOT NULL AND lon IS NOT NULL"
        ~ " AND NOT isnan(lat) AND NOT isnan(lon)),"
        ~ " j AS (SELECT l.name AS lname, l.label, d.street, d.city,"
        ~ " d.postcode, ST_Distance(l.g, ST_Point(d.lon, d.lat)) AS dist,"
        ~ " d.seq FROM locs l JOIN doors d ON ST_DWithin(l.g,"
        ~ " ST_Point(d.lon, d.lat), " ~ format("%.17g", rangeDeg) ~ ")),"
        ~ " best AS (SELECT lname, label, street, city, postcode, dist, seq,"
        ~ " row_number() OVER (PARTITION BY lname, street"
        ~ " ORDER BY dist, seq) AS rn FROM j),"
        ~ " capped AS (SELECT lname, label, street, city, postcode,"
        ~ " row_number() OVER (PARTITION BY street"
        ~ " ORDER BY dist, lname, label) AS cr FROM best WHERE rn = 1)"
        ~ " SELECT lname, label, street, city, postcode FROM capped"
        ~ " WHERE cr <= " ~ maxPerStreet.to!string
        ~ " ORDER BY lname, street");

    auto lnames = q.strings(0);
    auto labels = q.strings(1);
    auto streets = q.strings(2);
    auto cities = q.strings(3);
    auto postcodes = q.strings(4);

    PairRow[] out_;

    foreach (i; 0 .. lnames.length)
        out_ ~= PairRow(lnames[i], labels[i], streets[i],
            cities[i], postcodes[i]);

    return out_;
}

// ---- point-in-polygon over areas.geoparquet ----

// one AreaHit[] per query point (hits in area file order, the
// D grid's scan order); pts[i] = [lat, lon]
public AreaHit[][] containing(string dir, double[][] pts)
{
    if (pts.length == 0)
        return null;

    exec("CREATE OR REPLACE TEMP TABLE areas AS SELECT file_row_number AS seq,"
        ~ " osm_type, osm_id, name, label, admin_level, area, geometry"
        ~ " FROM read_parquet('" ~ escSql(dir ~ "/areas.geoparquet") ~ "',"
        ~ " file_row_number=true)");

    exec("CREATE INDEX areas_g_idx ON areas USING RTREE (geometry)");

    string[] vals;

    foreach (i, p; pts)
        vals ~= "(" ~ i.to!string ~ ", ST_Point("
            ~ format("%.17g", p[1]) ~ ", " ~ format("%.17g", p[0]) ~ "))";

    auto q = query(
        "SELECT q.idx, a.osm_type, a.osm_id, a.name, a.label,"
        ~ " a.admin_level, a.area FROM (VALUES " ~ vals.join(",") ~ ")"
        ~ " q(idx, g) JOIN areas a ON ST_Contains(a.geometry, q.g)"
        ~ " ORDER BY q.idx, a.seq");

    auto idxs = q.int64s(0);
    auto types = q.strings(1);
    auto ids = q.int64s(2);
    auto names = q.strings(3);
    auto labels = q.strings(4);
    auto admins = q.strings(5);
    auto m2s = q.doubles(6);

    auto out_ = new AreaHit[][](pts.length);

    foreach (i; 0 .. idxs.length)
    {
        enforce(idxs[i] >= 0 && cast(size_t) idxs[i] < pts.length,
            "containing row out of range");

        out_[idxs[i]] ~= AreaHit(types[i], ids[i], names[i],
            labels[i], admins[i], m2s[i]);
    }

    return out_;
}

// ---- fixtures ----

private enum string FIXTURE_DIR = "/tmp";

// anti-join + imputation + pair fixtures share one bucket dir
private void writeFixtures()
{
    // units/localities/cities/streets
    exec(`COPY (SELECT 'UG 06' AS name UNION ALL SELECT 'adarsh nagar'
               UNION ALL SELECT 'Mattanchery' UNION ALL SELECT 'Plot 1'
               UNION ALL SELECT 'VIJAYAWADA' UNION ALL SELECT NULL AS name)
           TO '` ~ FIXTURE_DIR ~ `/units.parquet' (FORMAT parquet)`);

    exec(`COPY (SELECT 'Adarsh Nagar' AS name UNION ALL SELECT 'Mattanchery' AS name)
           TO '` ~ FIXTURE_DIR ~ `/localities.parquet' (FORMAT parquet)`);

    exec(`COPY (SELECT 'Vijayawada' AS name UNION ALL SELECT 'Bangalore' AS name)
           TO '` ~ FIXTURE_DIR ~ `/cities.parquet' (FORMAT parquet)`);

    // streets.parquet: single-point geometries, the nearer one wins on
    // ties by file order; the two mislabel-shaped names ride along
    // for the roadNames anti-join test
    exec(`COPY (SELECT 'MG Road' AS name, ST_Point(77.60, 12.90) AS geometry
               UNION ALL SELECT '100 Feet Road', ST_Point(77.65, 12.95)
               UNION ALL SELECT 'Tie Road A', ST_Point(77.60, 12.902)
               UNION ALL SELECT 'Tie Road B', ST_Point(77.60, 12.898)
               UNION ALL SELECT 'Adarsh Nagar', ST_Point(77.6, 15.0)
               UNION ALL SELECT 'Mattanchery', ST_Point(77.6, 15.0)
               UNION ALL SELECT 'Kammanahalli Main Road', ST_Point(77.7, 13.0)
               UNION ALL SELECT 'L B S Colony', ST_Point(77.65, 12.95))
           TO '` ~ FIXTURE_DIR ~ `/streets.parquet' (FORMAT parquet)`);

    // addresses: in range of MG Road, out of range, no coords,
    // and one exactly on MG Road's point (distance 0 beats the
    // two tie roads either side of it)
    exec(`COPY (SELECT 12.9002 AS lat, 77.6002 AS lon, 'Bangalore' AS city,
               '560034' AS postcode
               UNION ALL SELECT 13.50, 77.60, '', ''
               UNION ALL SELECT NULL, NULL, '', ''
               UNION ALL SELECT 12.90, 77.60, 'Mysuru', '570001')
           TO '` ~ FIXTURE_DIR ~ `/addresses.parquet' (FORMAT parquet)`);
}

unittest
{
    import std.math : isNaN;

    writeFixtures();

    // anti-joins: null unit name behaves as "", kept
    assert(unitNames(FIXTURE_DIR) == ["UG 06", "Plot 1", ""]);

    assert(roadNames(FIXTURE_DIR) == ["100 Feet Road",
        "Kammanahalli Main Road", "L B S Colony", "MG Road",
        "Tie Road A", "Tie Road B"]);

    // imputation: nearest within 500 m; out-of-range and
    // coord-less rows get ""; a row on a road point gets that
    // road (distance 0)
    auto rDeg = 500.0 / DEG_TO_M;
    auto streets = imputeStreets(FIXTURE_DIR, 4, rDeg);

    assert(streets == ["MG Road", "", "", "MG Road"]);
}

unittest
{
    // pairs: one (locality, street) pair, nearest door rides its
    // city/postcode; out-of-range doors and disallowed streets
    // never pair
    writeFixtures();

    exec(`COPY (SELECT 'Kammanahalli' AS name, 'suburb' AS label,
               12.9001 AS lat, 77.60 AS lon)
           TO '` ~ FIXTURE_DIR ~ `/localities.parquet' (FORMAT parquet)`);

    bool[string] allowed;
    allowed["MG Road"] = true;
    allowed["Tie Road A"] = true;
    allowed["Tie Road B"] = true;

    auto pairs = pairList(FIXTURE_DIR, allowed,
        500.0 / DEG_TO_M, 800.0 / DEG_TO_M, 25);

    assert(pairs.length == 1);
    assert(pairs[0].locality == "Kammanahalli");
    assert(pairs[0].street == "MG Road");

    // nearest door won: the row on MG Road's point (12.90, 77.60)
    // is ~11 m from the locality, beating the 12.9002 row at ~25 m
    assert(pairs[0].city == "Mysuru");
    assert(pairs[0].postcode == "570001");
}

unittest
{
    import std.math : isNaN;

    writeFixtures();

    // anchors: per row the nearest point of its imputed street
    // within 800 m; no-coords and street-less rows get NaN
    auto rDeg = 500.0 / DEG_TO_M;
    auto aDeg = 800.0 / DEG_TO_M;
    auto anchors = roadAnchorByRow(FIXTURE_DIR, 4, rDeg, aDeg);

    assert(anchors["lat"][0] == 12.90 && anchors["lon"][0] == 77.60);
    assert(isNaN(anchors["lat"][1]) && isNaN(anchors["lon"][1]));
    assert(isNaN(anchors["lat"][2]));
    assert(anchors["lat"][3] == 12.90 && anchors["lon"][3] == 77.60);

    // pools: in-range names capped at k, nearest first, ties by
    // earlier streets row; an empty range yields no pool entry
    // (the caller falls back to the full vocab)
    exec(`COPY (SELECT 'Kammanahalli' AS name, 'suburb' AS label,
               12.9001 AS lat, 77.60 AS lon
               UNION ALL SELECT 'Far Away' AS name, 'suburb' AS label,
               13.50 AS lat, 77.60 AS lon)
           TO '` ~ FIXTURE_DIR ~ `/localities.parquet' (FORMAT parquet)`);

    auto pools = roadPoolByLocality(FIXTURE_DIR, aDeg, 2);

    assert(pools["Kammanahalli"] == ["MG Road", "Tie Road A"]);
    assert("Far Away" !in pools);
}

unittest
{
    import std.math : isNaN;

    writeFixtures();

    // locality pools per anchor: in-range capped at k, nearest
    // first (point-less localities excluded); an out-of-range
    // anchor gets an empty pool; a NaN anchor gets the full
    // vocab in file order
    exec(`COPY (SELECT 'Mid' AS name, 'suburb' AS label,
               12.945 AS lat, 77.60 AS lon
               UNION ALL SELECT 'Near', 'suburb', 12.956, 77.60
               UNION ALL SELECT 'Far', 'suburb', 12.94, 77.60
               UNION ALL SELECT 'NoPoint', 'suburb', NULL, NULL)
           TO '` ~ FIXTURE_DIR ~ `/localities.parquet' (FORMAT parquet)`);

    auto aDeg = 800.0 / DEG_TO_M;

    // 800 m ≈ 0.00719°: Mid (~556 m) and Near (~668 m) are in
    // range of 12.95, Far (~1113 m) is not; the 12.80 anchor is
    // out of range of everything
    auto pools = localityPoolByRow(FIXTURE_DIR, 0,
        [12.95, 12.80, double.nan], [77.60, 77.60, double.nan],
        aDeg, 50);

    assert(pools[0] == [LocRef("Mid", "suburb"), LocRef("Near", "suburb")]);
    assert(pools[1] == []);
    assert(pools[2] == [LocRef("Mid", "suburb"), LocRef("Near", "suburb"),
        LocRef("Far", "suburb"), LocRef("NoPoint", "suburb")]);

    // the k cap applies to the in-range pool too
    auto capped = localityPoolByRow(FIXTURE_DIR, 0,
        [12.95], [77.60], aDeg, 1);

    assert(capped[0] == [LocRef("Mid", "suburb")]);
}

unittest
{
    // containment: point in the square but not in its hole,
    // far point, hits in area file order
    import std.format : format;

    import bucket_generator.wkb : Ring, wkbMultiPolygon;

    auto outer = Ring([2.0, 2.0, 6.0, 6.0, 2.0], [2.0, 6.0, 6.0, 2.0, 2.0]);
    auto hole = Ring([3.0, 3.0, 5.0, 5.0, 3.0], [3.0, 5.0, 5.0, 3.0, 3.0]);
    auto block = wkbMultiPolygon([outer], [hole]);
    auto city = wkbMultiPolygon(
        [Ring([0.0, 0.0, 14.0, 14.0, 0.0], [0.0, 14.0, 14.0, 0.0, 0.0])],
        []);

    exec(`COPY (SELECT 'R' AS osm_type, 176 AS osm_id, 'Ward 176' AS name,
               'suburb' AS label, '10' AS admin_level, 16.0 AS area,
               ST_GeomFromWKB(unhex('` ~ format("%(%02x%)", block)
        ~ `')) AS geometry
               UNION ALL
               SELECT 'R', 2, 'Bengaluru', '', '8', 196.0,
               ST_GeomFromWKB(unhex('` ~ format("%(%02x%)", city) ~ `')))
           TO '` ~ FIXTURE_DIR ~ `/areas.geoparquet' (FORMAT parquet)`);

    auto hits = containing(FIXTURE_DIR,
        [[2.5, 3.0], [4.0, 4.0], [1.0, 13.0], [50.0, 50.0]]);

    // inside the ward, outside its hole: ward + city
    assert(hits[0].length == 2);
    assert(hits[0][0].osmId == 176);
    assert(hits[0][1].osmId == 2);

    // inside the hole: city only
    assert(hits[1].length == 1 && hits[1][0].osmId == 2);

    // inside the city only
    assert(hits[2].length == 1 && hits[2][0].osmId == 2);

    // outside everything
    assert(hits[3].length == 0);
}
