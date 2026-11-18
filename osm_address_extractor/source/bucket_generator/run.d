// bucket_generator.run: .osm.pbf → bucket parquets, all in SQL.
//
// The bucket_generator pipeline as one DuckDB database: the pbf
// decode, the gates (gates.sql macros) and every bucket select
// run as SQL through the shared engine core
// (address_generator.parquet_input); D only drives. Same engine
// and vendored spatial extension as stage 2.
//
//   ST_ReadOSM -> raw -> els -> way_pts -> way_geom, rel_geom
//                   |     |                 |          |
//                   |     |                 +-> kept   +-> areas
//                   |     |                     |
//                   |     +--> one windowed SELECT per bucket
//                   |             -> COPY parquet
//
// Every first-seen pick is an arg_min over the kept scan order
// (kord, id) = nodes, then ways, then relations, ids ascending,
// so bucket semantics (first contributor, first valid point,
// locality label upgrade, city head order) are windows, not a
// scan loop.
//
// usage: osm_address_extractor -i <pbf> -o <bucketdir>
module bucket_generator.run;

import std.array : join;
import std.conv : to;
import std.datetime.stopwatch : StopWatch, AutoStart;
import std.exception : enforce;
import std.file : mkdirRecurse;
import std.getopt : defaultGetoptPrinter, getopt, GetOptException,
    GetoptResult;
import std.stdio : stderr, writeln;

import address_generator.parquet_input : exec, query, escSql;

// gates.sql sits next to this file, embedded at compile time
private enum string GATES_SQL = import("bucket_generator/gates.sql");

// highway values naming roads: only these join the streets sidecar
private immutable ROADS = [
	"pedestrian", "unclassified", "footway", "residential",
	"tertiary", "secondary", "primary", "trunk", "motorway",
	"cycleway",
];

// place values naming localities or boundaries: never POIs
private immutable PLACE_TYPES = [
	"continent", "country", "state", "region", "province",
	"district", "county", "municipality", "city", "borough",
	"suburb", "quarter", "neighbourhood", "town", "village",
	"hamlet", "isolated_dwelling", "farm", "city_block",
	"square", "locality", "island", "islet",
];

// landuse values naming settlements or tracts: localities
private immutable AREA_LANDUSES = [
	"residential", "industrial", "construction", "farmland",
	"farmyard", "forest", "grass", "orchard", "quarry",
	"allotments", "commercial", "retail", "offices",
];

// place values naming settlements
private immutable SETTLEMENTS = [
	"suburb", "quarter", "neighbourhood", "city_block",
	"village", "hamlet", "town", "locality",
	"isolated_dwelling",
];

// name-variant columns feeding city/locality buckets
private immutable VARIANTS = ["alt_name", "old_name", "official_name", "short_name"];

// named landuse minimum area (m^2): smaller is a pair of
// buildings, not a neighbourhood (NULL never passes)
private enum LOCALITY_MIN_AREA_M2 = 40_469;

// addr:city is free-text junk: keep the top values only
private enum CITY_MAX = 500;

// level cap: "mezzanine" fits, junk phrases stay out
private enum MAX_LEVEL_CHARS = 12;

// a ring needs >= 4 pts (first = last)
private enum MIN_RING_PTS = 4;

// member roles that trace rings: '' means outer
private enum ROLES = "('', 'outer', 'inner')";

// the want filter: everything the gates could keep or record
private enum WANT = r"
    has_addr(tags)
    OR tag(tags, 'place') != ''
    OR tag(tags, 'highway') != ''
    OR tag(tags, 'landuse') != ''
    OR tag(tags, 'boundary') != ''
    OR tag(tags, 'admin_level') != ''
    OR (tag(tags, 'type') = 'boundary' AND tag(tags, 'name') != '')
";

// the folded pass runs only where a row can be an address
private enum ROW_WANT = "has_addr(tags) OR tag(tags, 'level') != ''";

// one-line header above the generated option list (--help)
private enum USAGE =
	"usage: osm_address_extractor -i <input.osm.pbf> -o <bucketdir>";

// ('a', 'b') list literal from a value tuple
private string inList(const(string)[] vals)
{
	string[] quoted;

	foreach (v; vals)
		quoted ~= "'" ~ v ~ "'";

	return "(" ~ quoted.join(", ") ~ ")";
}

private void copyParquet(string path, string select)
{
	exec("COPY (" ~ select ~ ") TO '" ~ escSql(path)
		~ "' (FORMAT parquet)");
}

// name-variant add events off the settled/city base CTE: the
// column value split on ';', trimmed, in column order
private string variantEvents(string cols)
{
	string[] parts;

	foreach (i, c; VARIANTS)
		parts ~= "SELECT " ~ cols ~ ", trim(seg) AS name, "
			~ (i + 1).to!string
			~ " AS b FROM settled, unnest(string_split(cleaned(tags, '"
			~ c ~ "'), ';')) AS u(seg)";

	return parts.join(" UNION ALL ");
}

// ---------------- substrate ----------------

private string rawSql(string pbf)
{
	return r"
        CREATE OR REPLACE TABLE raw AS
        SELECT kind::VARCHAR AS kind, id, tags, lat, lon,
               refs, ref_roles, ref_types
        FROM ST_ReadOSM('" ~ escSql(pbf) ~ r"')
        WHERE kind != 'changeset'
    ";
}

// els: keep() over the want rows, and the cleaning chain folded
// into one pass over the KEPT address rows. Non-kept rows are
// out: buckets read keep_flag, areas read is_area_relation
private string elsSql()
{
	return r"
    CREATE OR REPLACE TABLE els AS
    WITH
    want AS (
        SELECT kind, id, tags, lat, lon, keep(tags) AS keep_flag
        FROM raw
        WHERE " ~ WANT ~ r"
    ),
    clean AS (
        SELECT kind, id,
               strip_edge(clean_invisible(tag(tags, 'addr:housenumber'))) AS door,
               strip_edge(clean_invisible(tag(tags, 'addr:street'))) AS street,
               strip_edge(clean_invisible(tag(tags, 'addr:neighbourhood'))) AS neighbourhood,
               strip_edge(clean_invisible(tag(tags, 'addr:suburb'))) AS suburb,
               strip_edge(clean_invisible(tag(tags, 'addr:quarter'))) AS quarter,
               strip_edge(clean_invisible(tag(tags, 'addr:city'))) AS city,
               strip_edge(clean_invisible(tag(tags, 'addr:postcode'))) AS postcode,
               strip_edge(clean_invisible(tag(tags, 'addr:housename'))) AS housename,
               strip_edge(clean_invisible(tag(tags, 'addr:unit'))) AS unit,
               CASE WHEN strip_edge(clean_invisible(tag(tags, 'level'))) = ''
                    THEN strip_edge(clean_invisible(tag(tags, 'addr:floor')))
                    ELSE strip_edge(clean_invisible(tag(tags, 'level')))
               END AS level
        FROM want
        WHERE keep_flag AND (" ~ ROW_WANT ~ r")
    ),
    fin AS (
        SELECT kind, id,
               CASE WHEN value_reject(door) IN ('parens', 'phone', 'semicolon')
                    OR is_junk_door(door) THEN '' ELSE door END AS door,
               CASE WHEN value_reject(street) IS NOT NULL
                    OR is_direction_street(street)
                    THEN '' ELSE street END AS street,
               CASE WHEN value_reject(neighbourhood) IS NOT NULL
                    THEN '' ELSE neighbourhood END AS neighbourhood,
               CASE WHEN value_reject(suburb) IS NOT NULL
                    THEN '' ELSE suburb END AS suburb,
               CASE WHEN value_reject(quarter) IS NOT NULL
                    THEN '' ELSE quarter END AS quarter,
               CASE WHEN value_reject(city) IS NOT NULL
                    OR is_street_like_city(city)
                    THEN '' ELSE city END AS city,
               CASE WHEN value_reject(postcode) IS NOT NULL
                    OR (postcode != '' AND NOT is_postcode(postcode))
                    THEN '' ELSE postcode END AS postcode,
               CASE WHEN value_reject(housename) IS NOT NULL
                    OR is_direction_housename(housename)
                    THEN '' ELSE housename END AS housename,
               CASE WHEN value_reject(unit) IS NOT NULL
                    THEN '' ELSE unit END AS unit,
               CASE WHEN value_reject(level) IN ('parens', 'phone', 'comma')
                     OR (value_reject(level) = 'semicolon'
                         AND NOT is_level_pack(level))
                    THEN '' ELSE level END AS lvl
        FROM clean
    ),
    routed AS (
        SELECT kind, id, door, street, neighbourhood, suburb,
               quarter, city, postcode, housename,
               clean_unit(unit, lvl) AS cu, lvl
        FROM fin
    ),
    rf AS (
        SELECT r.kind, r.id,
               struct_pack(
                   door := r.door, street := r.street,
                   neighbourhood := r.neighbourhood, suburb := r.suburb,
                   quarter := r.quarter, city := r.city,
                   postcode := r.postcode, housename := r.housename,
                   unit := r.cu.unit,
                   level := CASE WHEN r.cu.level != '' THEN r.cu.level
                                ELSE r.lvl END,
                   is_address := len(list_filter(
                       [r.door, r.street, r.neighbourhood, r.suburb,
                        r.quarter, r.city, r.postcode, r.housename,
                        r.cu.unit,
                        CASE WHEN r.cu.level != '' THEN r.cu.level
                             ELSE r.lvl END],
                       lambda v: v != '')) > 0
               ) AS rf
        FROM routed r
    )
    SELECT w.kind, w.id, w.tags, w.lat, w.lon, w.keep_flag, x.rf
    FROM want w
    LEFT JOIN rf x ON x.kind = w.kind AND x.id = w.id
    WHERE w.keep_flag
       OR (w.kind = 'relation' AND is_area_relation(w.tags))
";
}

private enum REL_MEMBERS_SQL = r"
    CREATE OR REPLACE TABLE rel_members AS
    SELECT e.id AS rel_id, t.i + 1 AS mem_ord,
           r.refs[t.i + 1] AS way_id,
           coalesce(r.ref_roles[t.i + 1], '') AS role
    FROM els e
    JOIN raw r ON r.id = e.id AND r.kind = 'relation'
    CROSS JOIN range(coalesce(len(r.refs), 0)) AS t(i)
    WHERE e.kind = 'relation' AND r.ref_types[t.i + 1] = 'way'
";

// node points only for ways a bucket consumes: kept ways
// (way_geom lines, streets geometry) and relation member
// ways (rel_geom rings)
private enum WAY_PTS_SQL = r"
    CREATE OR REPLACE TABLE way_pts AS
    WITH want AS (
        SELECT id AS way_id FROM els
        WHERE kind = 'way' AND keep_flag
        UNION
        SELECT way_id FROM rel_members
    ),
    wn AS (
        SELECT w.id AS way_id, t.i + 1 AS node_ord,
               w.refs[t.i + 1] AS node_id
        FROM raw w
        JOIN want ON want.way_id = w.id
        CROSS JOIN range(coalesce(len(w.refs), 0)) AS t(i)
        WHERE w.kind = 'way'
    )
    SELECT wn.way_id, wn.node_ord, n.lat, n.lon
    FROM wn
    LEFT JOIN raw n ON n.id = wn.node_id AND n.kind = 'node'
";

// kept ways' geometry: closed lines become polygons, the rest
// keep the line centroid and area 0
private string wayGeomSql()
{
	return r"
    CREATE OR REPLACE TABLE way_geom AS
    WITH pts AS (
        SELECT wp.way_id, wp.node_ord, wp.lat, wp.lon
        FROM way_pts wp
        JOIN els e ON e.id = wp.way_id AND e.kind = 'way'
                   AND e.keep_flag
        WHERE wp.lat IS NOT NULL
    ),
    agg AS (
        SELECT way_id, list(ST_MakePoint(lon, lat) ORDER BY node_ord) AS pts
        FROM pts
        GROUP BY way_id
    ),
    lines AS (
        SELECT way_id, pts,
               len(pts) >= " ~ MIN_RING_PTS.to!string ~ r" AND pts[1] = pts[len(pts)] AS closed,
           CASE WHEN len(pts) >= 2
                THEN ST_MakeLine(pts::GEOMETRY[])
                ELSE pts[1]::GEOMETRY END AS g
        FROM agg
    )
    SELECT way_id,
           CASE WHEN closed
                THEN ST_Y(ST_Centroid(ST_MakePolygon(g)))
                ELSE ST_Y(ST_Centroid(g)) END AS lat,
           CASE WHEN closed
                THEN ST_X(ST_Centroid(ST_MakePolygon(g)))
                ELSE ST_X(ST_Centroid(g)) END AS lon,
           CASE WHEN closed
                THEN ST_Area_Spheroid(ST_MakePolygon(g))
                ELSE 0.0 END AS area
    FROM lines
";
}

// keep/area relations' geometry: member way lines -> area built
// by the extension. A relation whose no ring closes (clipped or
// degenerate) yields no row: no polygon, nothing to route or
// persist
private string relGeomSql()
{
	return r"
    CREATE OR REPLACE TABLE rel_geom AS
    WITH lines AS (
        SELECT rm.rel_id AS rel_id, rm.way_id AS way_id,
               arg_min(rm.role, rm.mem_ord) AS role,
               min(rm.mem_ord) AS mem_ord,
               ST_MakeLine(list(ST_MakePoint(wp.lon, wp.lat) ORDER BY wp.node_ord)
                           ::GEOMETRY[]) AS line
        FROM rel_members rm
        JOIN way_pts wp ON wp.way_id = rm.way_id
        WHERE wp.lat IS NOT NULL
        GROUP BY rm.rel_id, rm.way_id
        HAVING count(*) >= 2
    ),
    built AS (
        SELECT rel_id,
               ST_BuildArea(ST_Node(ST_Collect(list(line ORDER BY mem_ord,
                                                      way_id) FILTER (
                   WHERE role IN " ~ ROLES ~ r")))) AS b
        FROM lines
        GROUP BY rel_id
    )
    SELECT rel_id,
           ST_Y(ST_Centroid(b)) AS lat,
           ST_X(ST_Centroid(b)) AS lon,
           ST_Area_Spheroid(b) AS area,
           ST_AsWKB(ST_Multi(b)) AS wkb
    FROM built
    WHERE b IS NOT NULL AND NOT ST_IsEmpty(b)
";
}

// the kept rows with geometry and the scan order
private enum KEPT_SQL = r"
    CREATE OR REPLACE TABLE kept AS
    SELECT e.kind AS kind, e.id AS id, e.tags AS tags, e.rf AS rf,
           coalesce(e.lat, w.lat, r.lat) AS lat,
           coalesce(e.lon, w.lon, r.lon) AS lon,
           coalesce(w.area, r.area) AS area,
           CASE e.kind WHEN 'node' THEN 1 WHEN 'way' THEN 2
                       ELSE 3 END AS kord
    FROM els e
    LEFT JOIN way_geom w ON e.kind = 'way' AND w.way_id = e.id
    LEFT JOIN rel_geom r ON e.kind = 'relation' AND r.rel_id = e.id
    WHERE e.keep_flag
";

// ---------------- buckets ----------------

private string poisSql()
{
	return r"
    WITH base AS (
        SELECT kord, id, kind, place_of(tags) AS place,
               tag(tags, 'highway') AS highway,
               tag(tags, 'boundary') AS boundary,
               cleaned(tags, 'name') AS name
        FROM kept
    ),
    ev AS (
        SELECT kord, id, kind, name
        FROM base
        WHERE place NOT IN " ~ inList(PLACE_TYPES) ~ r"
          AND place NOT IN " ~ inList(AREA_LANDUSES) ~ r"
          AND (kind != 'way' OR highway NOT IN " ~ inList(ROADS) ~ r")
          AND boundary != 'postal_code'
          AND name_ok(name)
    )
    SELECT name, prov.t AS osm_type, prov.i AS osm_id
    FROM (
        SELECT name,
               arg_min(struct_pack(t := kind, i := id),
                       struct_pack(k := kord, i := id)) AS prov
        FROM ev GROUP BY name
    )
    ORDER BY name
";
}

// locality events: settlement names first-seen (label from the
// place type), their name variants, and big enough named area
// landuses. Per name: the first contributor, the first specific
// label (quarter/neighbourhood — suburb is the upgradeable
// default) and the first valid point. A table, not a COPY: the
// block-candidate bucket reuses the same deduped names
private string localitiesTableSql()
{
	return r"
    CREATE OR REPLACE TABLE locals AS
    WITH settled AS (
        SELECT kord, id, kind, lat, lon, tags,
               locality_label(place_of(tags)) AS label
        FROM kept
        WHERE place_of(tags) IN " ~ inList(SETTLEMENTS) ~ r"
    ),
    names AS (
        SELECT kord, id, kind, lat, lon, label,
               cleaned(tags, 'name') AS name, 0 AS b
        FROM settled
        UNION ALL " ~ variantEvents("kord, id, kind, lat, lon, label") ~ r"
    ),
    ev AS (
        SELECT kord, id, kind, lat, lon, label, name, b
        FROM names WHERE name_ok(name)
        UNION ALL
        SELECT kord, id, kind, lat, lon,
               'suburb', cleaned(tags, 'name'), 0
        FROM kept
        WHERE place_of(tags) IN " ~ inList(AREA_LANDUSES) ~ r"
          AND area >= " ~ LOCALITY_MIN_AREA_M2.to!string ~ r"
          AND name_ok(cleaned(tags, 'name'))
    )
    SELECT name,
           coalesce(lbl, 'suburb') AS label,
           prov.t AS osm_type,
           pt.lat, pt.lon, prov.i AS osm_id
    FROM (
        SELECT name,
               arg_min(struct_pack(t := kind, i := id),
                       struct_pack(k := kord, i := id, b := b)) AS prov,
               arg_min(label, struct_pack(k := kord, i := id, b := b))
                   FILTER (WHERE label != 'suburb') AS lbl,
               arg_min(struct_pack(lat := lat, lon := lon),
                       struct_pack(k := kord, i := id, b := b))
                   FILTER (WHERE lat IS NOT NULL) AS pt
        FROM ev GROUP BY name
    )
    ORDER BY name
";
}

// the block-candidate surface for stage 2's block→parent pairs:
// block-named area relations ("Jayanagar 4th Block") and
// bare-block locality points ("4th Block") as one bucket. An
// area's anchor (the stored centroid) must sit inside its own
// polygon — the check stage 2 used to run; a bare point cannot
// lie outside itself. Areas win the name dedupe: the polygon
// back beats a bare point. Localities keep their own bucket —
// range queries need the full point set
private string blockCandidatesSql()
{
	return r"
    WITH area_ev AS (
        SELECT e.kind, e.id, e.kord,
               tag(e.tags, 'name') AS name,
               r.lat, r.lon
        FROM kept e
        JOIN rel_geom r ON r.rel_id = e.id
        WHERE e.kind = 'relation' AND is_area_relation(e.tags)
          AND regexp_full_match(tag(e.tags, 'name'),
              '.*[[:space:]]+block[[:space:]]*', 'i')
          AND ST_Contains(ST_GeomFromWKB(r.wkb),
                          ST_Point(r.lon, r.lat))
    ),
    loc_ev AS (
        SELECT osm_type AS kind, osm_id AS id,
               CASE osm_type WHEN 'node' THEN 1
                             WHEN 'way' THEN 2
                             ELSE 3 END AS kord,
               name, lat, lon
        FROM locals
        WHERE regexp_full_match(name,
              '([^[:space:]]{1,2}|[0-9]+(st|nd|rd|th))'
                  || '[[:space:]]+block', 'i')
    ),
    ev AS (
        SELECT 0 AS src, * FROM area_ev
        UNION ALL
        SELECT 1 AS src, * FROM loc_ev
    )
    SELECT name, prov.t AS osm_type, prov.i AS osm_id,
           pt.lat, pt.lon
    FROM (
        SELECT name,
               arg_min(struct_pack(t := kind, i := id),
                       struct_pack(s := src, k := kord,
                                   i := id)) AS prov,
               arg_min(struct_pack(lat := lat, lon := lon),
                       struct_pack(s := src, k := kord,
                                   i := id))
                   FILTER (WHERE lat IS NOT NULL) AS pt
        FROM ev
        GROUP BY name
    )
    WHERE pt.lat IS NOT NULL AND pt.lon IS NOT NULL
    ORDER BY name
";
}

// cities: place=city names and their variants first (in
// first-seen order), then the addr:city values by frequency,
// capped at CITY_MAX total. Tail values already in the head
// stay counted for frequency but emit once
private string citiesSql()
{
	return r"
    WITH settled AS (
        SELECT kord, id, kind, tags
        FROM kept
        WHERE place_of(tags) = 'city'
    ),
    names AS (
        SELECT kord, id, kind, cleaned(tags, 'name') AS name, 0 AS b
        FROM settled
        UNION ALL " ~ variantEvents("kord, id, kind") ~ r"
    ),
    head AS (
        SELECT name,
               arg_min(struct_pack(t := kind, i := id),
                       struct_pack(k := kord, i := id, b := b)) AS prov,
               min(struct_pack(k := kord, i := id, b := b)) AS seq
        FROM names WHERE name_ok(name) GROUP BY name
    ),
    tail AS (
        SELECT name, count(*) AS cnt,
               arg_min(struct_pack(t := kind, i := id),
                       struct_pack(k := kord, i := id)) AS prov
        FROM (
            SELECT kord, id, kind,
                   cleaned(tags, 'addr:city') AS name
            FROM kept
        )
        WHERE name_ok(name)
        GROUP BY name
    ),
    head_rows AS (
        SELECT name, prov.t AS osm_type, prov.i AS osm_id,
               0 AS src,
               row_number() OVER (ORDER BY seq) AS ord
        FROM head
    ),
    tail_rows AS (
        SELECT name, prov.t AS osm_type, prov.i AS osm_id,
               1 AS src,
               row_number() OVER (ORDER BY cnt DESC, name) AS ord
        FROM tail
        WHERE name NOT IN (SELECT name FROM head)
        QUALIFY ord <= " ~ CITY_MAX.to!string ~ r" - (SELECT count(*) FROM head)
    )
    SELECT name, osm_type, osm_id
    FROM (
        SELECT * FROM head_rows UNION ALL SELECT * FROM tail_rows
    )
    ORDER BY src, ord
";
}

// 6-byte all-digit values: the D gate compares byte length, a
// 6-char Kannada-digit PIN is 18 bytes and drops
private enum POSTCODES_SQL = r"
    WITH ev AS (
        SELECT kord, id, kind, cleaned(tags, 'addr:postcode') AS v
        FROM kept
        UNION ALL
        SELECT kord, id, kind, cleaned(tags, 'postal_code') AS v
        FROM kept
        WHERE tag(tags, 'boundary') = 'postal_code'
    )
    SELECT v AS name, prov.t AS osm_type, prov.i AS osm_id
    FROM (
        SELECT v,
               arg_min(struct_pack(t := kind, i := id),
                       struct_pack(k := kord, i := id)) AS prov
        FROM ev
        WHERE octet_length(encode(v)) = 6
          AND regexp_full_match(v, '[\\p{Nd}\\p{Nl}\\p{No}]+')
        GROUP BY v
    )
    ORDER BY name
";

// a leading direction marker makes a housename a proximity
// description: it reports, never joins. The tag gate first —
// no addr:housename, no name to test
private enum HOUSENAMES_SQL = r"
    WITH ev AS (
        SELECT kord, id, kind, cleaned(tags, 'addr:housename') AS name
        FROM kept
        WHERE tag(tags, 'addr:housename') != ''
          AND NOT is_direction_housename(cleaned(tags, 'addr:housename'))
          AND name_ok(cleaned(tags, 'addr:housename'))
    )
    SELECT name, prov.t AS osm_type, prov.i AS osm_id
    FROM (
        SELECT name,
               arg_min(struct_pack(t := kind, i := id),
                       struct_pack(k := kord, i := id)) AS prov
        FROM ev GROUP BY name
    )
    ORDER BY name
";

// the unit vocab takes shapes, not names: clean_unit already
// gated the shape, the name policy would reject digit-only
// codes like "2" or "B1". The tag gate rides first — no tag,
// no unit — and MATERIALIZED holds it before clean_unit, plain
// CTE filters get pushed past the cleaning regexes
private enum UNITS_SQL = r"
    WITH base AS MATERIALIZED (
        SELECT kord, id, kind,
               clean_unit(cleaned(tags, 'addr:unit'),
                          cleaned(tags, 'level')) AS cu
        FROM kept
        WHERE tag(tags, 'addr:unit') != ''
    ),
    ev AS (
        SELECT kord, id, kind, cu.unit AS name
        FROM base
        WHERE cu.unit != ''
          AND name_reject(cu.unit) IS NULL
    )
    SELECT name, prov.t AS osm_type, prov.i AS osm_id
    FROM (
        SELECT name,
               arg_min(struct_pack(t := kind, i := id),
                       struct_pack(k := kord, i := id)) AS prov
        FROM ev GROUP BY name
    )
    ORDER BY name
";

// the level value: level=*, a floor phrase misfiled into
// addr:unit, addr:floor; semicolon packs split, every segment
// must be short and comma-free. Tag gate + MATERIALIZED as in
// units: the coalesce regexes ride only on tagged rows
private string levelsSql()
{
	return r"
    WITH base AS MATERIALIZED (
        SELECT kord, id, kind, coalesce(
            nullif(cleaned(tags, 'level'), ''),
            nullif(clean_unit(cleaned(tags, 'addr:unit'),
                              cleaned(tags, 'level')).level, ''),
            nullif(cleaned(tags, 'addr:floor'), '')
        ) AS lvl
        FROM kept
        WHERE tag(tags, 'level') != ''
           OR tag(tags, 'addr:unit') != ''
           OR tag(tags, 'addr:floor') != ''
    ),
    ev0 AS (
        SELECT kord, id, kind, trim(seg) AS name
        FROM base, unnest(string_split(lvl, ';')) AS u(seg)
    ),
    ev AS (
        SELECT * FROM ev0
        WHERE name != ''
          AND len(name) <= " ~ MAX_LEVEL_CHARS.to!string ~ r"
          AND NOT contains(name, ',')
    )
    SELECT name, prov.t AS osm_type, prov.i AS osm_id
    FROM (
        SELECT name,
               arg_min(struct_pack(t := kind, i := id),
                       struct_pack(k := kord, i := id)) AS prov
        FROM ev GROUP BY name
    )
    ORDER BY name
";
}

private enum ADDRESSES_SQL = r"
    SELECT kind AS osm_type, id AS osm_id, lat, lon,
           nullif(rf.door, '') AS door,
           nullif(rf.street, '') AS street,
           nullif(rf.neighbourhood, '') AS neighbourhood,
           nullif(rf.suburb, '') AS suburb,
           nullif(rf.quarter, '') AS quarter,
           nullif(rf.city, '') AS city,
           nullif(rf.postcode, '') AS postcode,
           nullif(rf.housename, '') AS housename,
           nullif(rf.unit, '') AS unit,
           nullif(rf.level, '') AS level
    FROM kept
    WHERE rf.is_address
    ORDER BY kord, id
";

// areas sidecar: named place/boundary relations with their
// built multipolygon; relations without one yield no row
private enum AREAS_SQL = r"
    SELECT e.kind AS osm_type, e.id AS osm_id,
           tag(e.tags, 'name') AS name,
           nullif(tag(e.tags, 'place'), '') AS label,
           nullif(tag(e.tags, 'admin_level'), '') AS admin_level,
           r.lat, r.lon, r.area, ST_GeomFromWKB(r.wkb) AS geometry
    FROM els e
    JOIN rel_geom r ON r.rel_id = e.id
    WHERE e.kind = 'relation' AND is_area_relation(e.tags)
    ORDER BY e.id
";

// streets sidecar: one row per named road way with its line
// geometry (single-node ways keep the point) — the nearest-road
// queries run ST_DWithin/ST_ClosestPoint against it directly
private string streetsSql()
{
	return r"
    WITH road_base AS (
        SELECT id AS way_id, cleaned(tags, 'name') AS name
        FROM kept
        WHERE kind = 'way'
          AND tag(tags, 'highway') IN " ~ inList(ROADS) ~ r"
    ),
    roads AS (
        SELECT * FROM road_base
        WHERE name_ok(name) AND NOT is_direction_street(name)
    ),
    agg AS (
        SELECT wp.way_id,
               list(ST_MakePoint(wp.lon, wp.lat) ORDER BY wp.node_ord)
                   AS pts
        FROM way_pts wp
        JOIN roads r ON r.way_id = wp.way_id
        WHERE wp.lat IS NOT NULL
        GROUP BY wp.way_id
    ),
    lines AS (
        SELECT r.way_id, r.name,
               CASE WHEN len(a.pts) >= 2
                    THEN ST_MakeLine(a.pts::GEOMETRY[])
                    ELSE a.pts[1]::GEOMETRY END AS geometry
        FROM roads r
        JOIN agg a ON a.way_id = r.way_id
    )
    SELECT name, geometry FROM lines
    ORDER BY way_id
";
}

// ---------------- pipeline ----------------

private struct Stats
{
	size_t nodes;
	size_t ways;
	size_t relations;

	size_t rows()
	{
		return nodes + ways + relations;
	}
}

// build the substrate, write the 11 files, report the kept
// counts
private Stats extract(string pbf, string outDir)
{
	// geometries are lon/lat (x/y): the spheroid functions
	// reject the lat/lon default axis order
	exec("SET geometry_always_xy = true");

	exec(GATES_SQL);

	foreach (sql; [rawSql(pbf), elsSql(), REL_MEMBERS_SQL,
			WAY_PTS_SQL, wayGeomSql(), relGeomSql(), KEPT_SQL])
		exec(sql);

	copyParquet(outDir ~ "/addresses.parquet", ADDRESSES_SQL);
	copyParquet(outDir ~ "/pois.parquet", poisSql());
	exec(localitiesTableSql());
	copyParquet(outDir ~ "/localities.parquet",
		"SELECT name, label, osm_type, lat, lon, osm_id"
		~ " FROM locals");
	copyParquet(outDir ~ "/block_candidates.parquet",
		blockCandidatesSql());
	copyParquet(outDir ~ "/cities.parquet", citiesSql());
	copyParquet(outDir ~ "/postcodes.parquet", POSTCODES_SQL);
	copyParquet(outDir ~ "/housenames.parquet", HOUSENAMES_SQL);
	copyParquet(outDir ~ "/units.parquet", UNITS_SQL);
	copyParquet(outDir ~ "/levels.parquet", levelsSql());
	copyParquet(outDir ~ "/streets.parquet", streetsSql());
	copyParquet(outDir ~ "/areas.geoparquet", AREAS_SQL);

	auto q = query(
		"SELECT count(*) FILTER (kind = 'node') AS n,"
		~ " count(*) FILTER (kind = 'way') AS w,"
		~ " count(*) FILTER (kind = 'relation') AS r FROM kept");

	Stats s;
	s.nodes = cast(size_t) q.int64s(0)[0];
	s.ways = cast(size_t) q.int64s(1)[0];
	s.relations = cast(size_t) q.int64s(2)[0];

	return s;
}

// ---------------- cli ----------------

private struct Opts
{
	string input;
	string outPath;
}

// both flags and output live here: the stage-1 binary's main
int main(string[] args)
{
	Opts o;
	GetoptResult helpInfo;

	try
	{
		helpInfo = getopt(args,
			"input|i", "osm pbf extract", &o.input,
			"out|o", "bucket output .parquet dir", &o.outPath);
	}
	catch (GetOptException e)
	{
		stderr.writefln("error: %s (try osm_address_extractor --help)", e.msg);
		return 1;
	}

	if (helpInfo.helpWanted)
	{
		defaultGetoptPrinter(USAGE, helpInfo.options);
		return 0;
	}

	enforce(o.input.length > 0, "missing -i");
	enforce(o.outPath.length > 0, "missing -o");

	mkdirRecurse(o.outPath);

	auto sw = StopWatch(AutoStart.yes);
	auto stats = extract(o.input, o.outPath);

	stderr.writefln("done: %s elements (nodes %s, ways %s, relations %s) in %.1fs",
		stats.rows, stats.nodes, stats.ways, stats.relations,
		sw.peek.total!"msecs" / 1e3);

	return 0;
}

unittest
{
	import std.exception : assertThrown;

	// --help short-circuits before the -i/-o checks
	assert(main(["osm_address_extractor", "--help"]) == 0);

	// unknown flags error instead of throwing to the runtime
	assert(main(["osm_address_extractor", "--nope"]) == 1);

	// missing -i fails loudly
	assertThrown(main(["osm_address_extractor"]));
}

unittest
{
	// block candidates: block-named area relations only when the
	// polygon centroid sits inside the polygon (an L-shaped ring's
	// centroid escapes its notch), bare-block localities; per name
	// one row, the area winning over a same-named locality point
	import std.format : format;
	import std.math : abs;

	exec("SET geometry_always_xy = true");
	exec(GATES_SQL);

	exec(r"
        CREATE OR REPLACE TABLE kept AS
        SELECT * FROM (VALUES
            ('relation', 1,
             map(['name'], ['Jayanagar']),
             NULL, 12.91, 77.60, 400.0, 3),
            ('relation', 2,
             map(['name', 'place'],
                 ['Jayanagar 4th Block', 'quarter']),
             NULL, 12.9075, 77.5925, 50.0, 3),
            ('relation', 3,
             map(['name', 'place'], ['L Shaped Block', 'quarter']),
             NULL, 1.3571, 1.3571, 7.0, 3)
        ) t(kind, id, tags, rf, lat, lon, area, kord)
    ");

	// the candidate polygons with their built centroid — the same
	// lat/lon stage 2 used to receive via the areas sidecar
	exec(r"
        CREATE OR REPLACE TABLE rel_geom AS
        WITH g AS (
            SELECT 1 AS rel_id,
                   ST_MakePolygon(ST_MakeLine([
                       ST_MakePoint(77.58, 12.90),
                       ST_MakePoint(77.62, 12.90),
                       ST_MakePoint(77.62, 12.92),
                       ST_MakePoint(77.58, 12.92),
                       ST_MakePoint(77.58, 12.90)]::GEOMETRY[]))
                   AS geom
            UNION ALL
            SELECT 2,
                   ST_MakePolygon(ST_MakeLine([
                       ST_MakePoint(77.585, 12.905),
                       ST_MakePoint(77.60, 12.905),
                       ST_MakePoint(77.60, 12.91),
                       ST_MakePoint(77.585, 12.91),
                       ST_MakePoint(77.585, 12.905)]::GEOMETRY[]))
            UNION ALL
            SELECT 3,
                   ST_MakePolygon(ST_MakeLine([
                       ST_MakePoint(0.0, 0.0),
                       ST_MakePoint(4.0, 0.0),
                       ST_MakePoint(4.0, 1.0),
                       ST_MakePoint(1.0, 1.0),
                       ST_MakePoint(1.0, 4.0),
                       ST_MakePoint(0.0, 4.0),
                       ST_MakePoint(0.0, 0.0)]::GEOMETRY[]))
        )
        SELECT rel_id,
               ST_Y(ST_Centroid(geom)) AS lat,
               ST_X(ST_Centroid(geom)) AS lon,
               0.0 AS area,
               ST_AsWKB(ST_Multi(geom)) AS wkb
        FROM g
    ");

	exec(r"
        CREATE OR REPLACE TABLE locals AS
        SELECT * FROM (VALUES
            ('4th Block', 'neighbourhood', 'node',
             12.909, 77.59, 10),
            ('Jayanagar 4th Block', 'quarter', 'node',
             12.9081, 77.5921, 11)
        ) t(name, label, osm_type, lat, lon, osm_id)
    ");

	auto q = query(blockCandidatesSql());

	auto names = q.strings(0);
	auto types = q.strings(1);
	auto ids = q.int64s(2);
	auto lats = q.doubles(3);
	auto lons = q.doubles(4);

	assert(names.length == 2, format("%s", names));

	// the bare-block locality survives on the bare shape
	assert(names[0] == "4th Block" && types[0] == "node");
	assert(ids[0] == 10);
	assert(abs(lats[0] - 12.909) < 1e-9);

	// the area relation wins the same-name dedupe; the polygon
	// centroid anchors the pair, never the locality point
	assert(names[1] == "Jayanagar 4th Block"
		&& types[1] == "relation");
	assert(ids[1] == 2);
	assert(abs(lats[1] - 12.9075) < 1e-9);
	assert(abs(lons[1] - 77.5925) < 1e-9);
}
