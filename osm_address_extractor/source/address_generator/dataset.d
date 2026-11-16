// address_generator: bucket parquets → BIO JSONL.
//
// <bucketdir>/{addresses,streets,pois,localities,cities,
//              postcodes}.parquet
//   │
//   ▼
// fill: addresses row → Components (+ vocab locality,
//       train-only dropout on droppable fields)
//   │
//   ▼
// compose (address_generator.grammar): components → tagged example
//   │
//   ▼
// train/test JSONL (deterministic md5 split on osm_type/osm_id;
// synthetic rows are train-only)
//
// Extraction (bucket_generator) already cleaned the units: junk doors
// blanked, separators stripped, vocab values filtered.
module address_generator.dataset;

import core.memory : GC;
import std.algorithm.comparison : min;
import std.algorithm.searching : canFind;
import std.algorithm.sorting : sort;
import std.array : split;
import std.conv : to;
import std.digest.md : md5Of;
import std.exception : enforce;
import std.file : exists, readText;
import std.format : format;
import std.getopt : defaultGetoptPrinter, getopt, GetOptException,
    GetoptResult;
import std.json : parseJSON;
import std.math : abs, isNaN;
import std.stdio : File, stderr, writeln;
import std.string : toLower;
import std.uni : isNumber;

import address_generator.coords : DEG_TO_M;
import address_generator.gates : isLocalityStreet, isOrdinalStreet;
import address_generator.buckets : Buckets, Locality, Road,
    loadBuckets;
import address_generator.grammar : Components, Example, Rng, Seg,
    compose, doorSegs, dottedInitials, gluedInitials, hasInitialRun,
    normalizeDots, restyleInitials, restyleDoor, synthRoad, words;
import address_generator.parquet_input : Parquet, load, loadNums,
    loadText, rowCount, exec, resetDb;
import address_generator.queries : AreaHit, LocRef, PairRow, containing,
    imputeStreets, localityPoolByRow, pairList, roadAnchorByRow,
    roadPoolByLocality;

// addresses column → BIO label; droppable fields may be
// dropped by train-only noise augmentation. quarter and
// neighbourhood merge into suburb (libpostal alignment)
private struct Field
{
    string col;
    string label;
    bool droppable;
}

private enum Field[] FIELDS = [
    Field("door", "house_number", false),
    Field("neighbourhood", "suburb", true),
    Field("suburb", "suburb", true),
    Field("quarter", "suburb", true),
    Field("city", "city", false),
    Field("postcode", "postcode", false),
];

// train-only noise: droppable field dropout
private enum P_DROPOUT = 0.15;

// train-only street dropout: trade and hand-eval addresses
// carry no road ("26/1, ORION MALL, INDUSTRIAL SUBURB, I
// BLOCK, RAJAJINAGAR, 560010") and the tagger must learn to
// emit no road at all — it hallucinated the block as road.
// Teaching rate, far above the measured 0.04% empty trade
// streets
private enum P_NO_ROAD = 0.05;

// train-only city dropout: trade input carries no city
// column, so the last locality sits before the PIN —
// without teaching that frame the model city-tags the
// trailing locality (impute hand50: RAJAJINAGAR,
// Mallathahalli). Teaching rate, far above the trade
// shape's OSM frequency (measured: multi-locality cityless
// rows are 4.2k vs 148k city-before-PIN)
private enum P_NO_CITY = 0.20;

// trade-license frequencies (NOTES.md): trade addresses
// almost always carry a locality between the road and city
private enum P_LOCALITY = 0.25;

// synthetic locality examples: real street names half the
// time, city and postcode usually present
private enum P_SYNTH_REAL_ROAD = 0.50;
private enum P_SYNTH_CITY = 0.85;
private enum P_SYNTH_POSTCODE = 0.85;
private enum DOOR_SYNTH_MAX = 9999;

// locality span floor (open 2 / Fix D): the median vocab
// locality renders exactly once (its synth row); rare names
// sit at the decision margin regardless of casing (GOTTIGERE
// 1 span absorbed into road, HULIMAVU 17 flaky, Kothnur 372
// solid). synthLocalities emits extra draws for names the
// real rows rendered fewer than FLOOR times. Pair rows run
// later and overshoot the floor benignly
private enum LOCALITY_SPAN_FLOOR = 4;

// nearest-point anchoring (libpostal's no-polygon fallback,
// NOTES 2026-10-09): injected localities come from the street's
// neighborhood, synthetic rows pick a street near the locality,
// so road and locality co-occur realistically. The pool is the
// localities within ANCHOR_RANGE_M of the anchor point, capped
// at the ANCHOR_K nearest, computed in DuckDB
// (queries.localityPoolByRow); an empty range yields an empty
// pool and the row gets no injected locality. Uniform sampling
// from the full vocab when no point is known
private enum ANCHOR_K = 50;

// rows per locality-pool query chunk — materializing rows ×
// ANCHOR_K pools at once would not fit
private enum POOL_CHUNK = 4096;

// ~meters per degree of latitude; double keeps the range
// ratio out of integer division
private enum ANCHOR_RANGE_M = 800;
private enum double ANCHOR_RANGE_DEG = ANCHOR_RANGE_M / DEG_TO_M;

// wholesale street imputation radius (NOTES 2026-10-26):
// addr:street is bulk-import junk too often (40.9% of rows
// carry a value no named road way matches); every real row's
// street is the nearest named-road geometry instead. 500 m
// is the coverage knee — 90% of address rows have a road
// point in range, rows beyond it lose the street
private enum IMPUTE_MAX_M = 500;
private enum double IMPUTE_MAX_DEG = IMPUTE_MAX_M / DEG_TO_M;

// (locality, street) coverage pairs cap: name-collapsed street
// vocabs pair one name with hundreds of localities ("2nd
// Cross Road" × 816 measured); only the nearest pairs are
// real adjacencies, the rest is oversampling noise
private enum PAIR_MAX_PER_STREET = 25;

// 80-20 split on the hash; synthetic locality rows are
// train-only, so real training rows are ~80% of the total
private enum TEST_SPLIT_PCT = 20;
private enum MIN_COMPS = 2;
private enum MIN_TOKENS = 3;

private enum DEFAULT_SEED = 42;

private struct Opts
{
    string inputFile;
    string train = "data/train.jsonl";
    string test = "data/test.jsonl";
    string excludeIds;
    int seed = DEFAULT_SEED;
}

private struct Stats
{
    size_t train;
    size_t test;
    size_t landmark;
    size_t multiRoad;
    size_t locality;
    size_t synth;
    size_t pair;
    size_t block;
}

// one-line header above the generated option list (--help)
private enum USAGE = "usage: osm_address_extractor-dataset -i <bucketdir>";

// stage-2 binary's main
int main(string[] args)
{
    Opts o;
    GetoptResult helpInfo;

    try
    {
        helpInfo = getopt(args,
            "input|i", "bucket output dir from stage 1", &o.inputFile,
            "train", "train jsonl (default: data/train.jsonl)", &o.train,
            "test", "test jsonl (default: data/test.jsonl)", &o.test,
            "exclude-ids", "osm_id records json to skip (eval examples)",
                &o.excludeIds,
            "seed", format!"rng seed (default: %s)"(DEFAULT_SEED), &o.seed);
    }
    catch (GetOptException e)
    {
        stderr.writefln("error: %s (try osm_address_extractor-dataset --help)",
            e.msg);
        return 1;
    }

    if (helpInfo.helpWanted)
    {
        defaultGetoptPrinter(USAGE, helpInfo.options);
        return 0;
    }

    enforce(o.inputFile.length > 0, "missing -i/--input-file");

    write(o);

    return 0;
}

unittest
{
    import std.exception : assertThrown;

    // --help short-circuits before the -i check
    assert(main(["osm_address_extractor-dataset", "--help"]) == 0);

    // unknown flags error instead of throwing to the runtime
    assert(main(["osm_address_extractor-dataset", "--nope"]) == 1);

    // missing -i fails loudly
    assertThrown(main(["osm_address_extractor-dataset"]));
}

private void write(Opts o)
{
    auto buckets = loadBuckets(o.inputFile);
    auto excluded = loadExcluded(o.excludeIds);

    string[] cols;

    foreach (f; FIELDS)
        cols ~= f.col;

    cols ~= ["housename", "unit", "level"];

    // wholesale street imputation: one road per row from the
    // streets sidecar — the addr:street column is never read
    // (not even loaded)
    auto addrPath = o.inputFile ~ "/addresses.parquet";
    auto nRows = rowCount(addrPath);

    auto streets = imputeStreets(o.inputFile, nRows,
        IMPUTE_MAX_DEG);

    // real way geometry for the spatial anchors, all in DuckDB:
    // per row the nearest point of its imputed street (the
    // injectLocal anchor), per locality the road names with a
    // way point in range (the synth anchor pool)
    auto anchors = roadAnchorByRow(o.inputFile, nRows,
        IMPUTE_MAX_DEG, ANCHOR_RANGE_DEG);
    auto pools = roadPoolByLocality(o.inputFile,
        ANCHOR_RANGE_DEG, ANCHOR_K);

    // the spatial phase leaves the DuckDB buffer pool pinned at
    // the memory cap and starves the per-chunk locality-pool
    // queries (OOM 2026-11-12); reset frees the whole engine,
    // the collect returns the freed D heap to the OS
    resetDb();
    GC.collect();

    auto rng = Rng(cast(uint) o.seed);
    auto drop = streetDropSet(buckets);
    Stats stats;

    // suburb spans per name, train real rows — the locality
    // frequency floor's input
    int[string] spanCounts;

    auto train = File(o.train, "w");
    auto test = File(o.test, "w");

    // address rows and locality pools stream in aligned chunks —
    // the full text column set never sits resident in the D heap
    for (size_t lo = 0; lo < nRows; lo += POOL_CHUNK)
    {
        auto hi = min(lo + POOL_CHUNK, nRows);

        auto pq = load(addrPath, cols, lo, hi);

        auto locPools = localityPoolByRow(o.inputFile, lo,
            anchors["lat"][lo .. hi], anchors["lon"][lo .. hi],
            ANCHOR_RANGE_DEG, ANCHOR_K);

        foreach (r; 0 .. pq.osmIds.length)
        {
            if (pq.osmIds[r] in excluded)
                continue;

            auto isTest = hashPct(pq.osmTypes[r], pq.osmIds[r]) < TEST_SPLIT_PCT;

            auto c = rowComponents(pq, r, &rng, !isTest, drop,
                streets[lo + r]);

            if (!minComps(c))
                continue;

            bool injected = injectLocal(c, &rng, buckets,
                locPools[r]);

            auto ex = compose(c, &rng, buckets, !isTest);

            if (ex.tokens.length < MIN_TOKENS)
                continue;

            if (!isTest)
                noteSpans(ex, spanCounts);

            if (injected)
                stats.locality++;

            // measured on the output itself: two road spans make a
            // multi-road street, a landmark tag a landmark row
            stats.multiRoad += spans(ex.tags, "road") >= 2;
            stats.landmark += spans(ex.tags, "landmark") >= 1;

            if (isTest)
            {
                test.writeln(jsonLine(ex));
                stats.test++;
            }
            else
            {
                train.writeln(jsonLine(ex));
                stats.train++;
            }
        }
    }

    // one synthetic example per distinct locality, train-only:
    // sz-test stays real OSM rows, its locality coverage comes
    // from the injected vocab localities. Rare names draw up to
    // the frequency floor
    stats.synth = synthLocalities(buckets, pools, &rng, train,
        spanCounts);

    // every (locality, street-with-a-door-nearby) pair, one
    // train-only row each: coverage the injection dice cannot
    // miss. Allowed streets: the road vocab minus mislabels
    // (drop set) and ordinal-only names
    bool[string] allowed;

    foreach (rd; buckets.roads)
        if (!isOrdinalStreet(rd.name)
            && rd.name.toLower() !in drop)
            allowed[rd.name] = true;

    stats.pair = pairRows(
        pairList(o.inputFile, allowed, IMPUTE_MAX_DEG,
            ANCHOR_RANGE_DEG, PAIR_MAX_PER_STREET),
        buckets, &rng, train);

    // block→parent pairs ("4th Block, Jayanagar"): one
    // block_candidates.parquet row per distinct block name;
    // missing file emits nothing
    stats.block = blockRows(blockList(o.inputFile),
        buckets, pools, &rng, train);

    // the pair enumeration leaves the GC heap at
    // its high-water mark; nothing after this point allocates much
    GC.collect();

    train.close();
    test.close();

    writeln(format!(
        `{"train": %s, "test": %s, "landmark": %s, "multi_road": %s, "locality": %s, "synth_locality": %s, "pair": %s, "block": %s}`)
        (stats.train, stats.test, stats.landmark, stats.multiRoad,
            stats.locality, stats.synth, stats.pair, stats.block));
}

// addresses row → components: plain numeric doors re-render to
// the trade mix, droppable locality fields may drop out
// (train-only). The street arrives imputed (wholesale,
// NOTES 2026-10-26) — the addr:street column is never read
private Components rowComponents(Parquet pq, size_t r, Rng* rng,
    bool noise, const bool[string] drop = null,
    string street = "")
{
    Components c;

    // door decorations stay untagged: the model must emit
    // the clean number downstream geocoders match on
    auto door = pq.texts["door"][r];

    if (door.length > 0)
    {
        if (isAllDigits(door))
            door = restyleDoor(door, rng);

        c.door = doorSegs(door);
    }

    c.road = street;

    // mislabeled street values blank like bucket_generator junk gates:
    // same-name locality/city or locality-shaped (NOTES open 2);
    // the row survives on its other fields
    if (c.road.length > 0 && (isLocalityStreet(c.road)
        || c.road.toLower() in drop))
        c.road = "";

    // street dropout: absent road is an outcome the model must
    // see, not a rendering defect to repair
    if (noise && c.road.length > 0 && rng.unit() < P_NO_ROAD)
        c.road = "";

    foreach (f; FIELDS)
    {
        if (!f.droppable)
            continue;

        auto val = pq.texts[f.col][r];

        if (val.length == 0)
            continue;

        if (noise && rng.unit() < P_DROPOUT)
            continue;

        c.localities ~= Locality(restyleInitials(rng, val),
            f.label);
    }

    c.city = pq.texts["city"][r];

    // city dropout: the last locality must be able to sit
    // before the PIN without sliding into the city slot
    if (noise)
        dropCity(c.city, rng);

    c.postcode = pq.texts["postcode"][r];

    // premises detail; the bare level=* number renders to
    // the floor dialect in the grammar
    c.house = pq.texts["housename"][r];
    c.unit = pq.texts["unit"][r];
    c.level = pq.texts["level"][r];

    return c;
}

// city dropout shared by the real, synth and pair row
// paths (see P_NO_CITY)
private void dropCity(ref string city, Rng* rng)
{
    if (city.length > 0 && rng.unit() < P_NO_CITY)
        city = "";
}

// vocab locality joins the row's own in trade position; runs
// after the MIN_COMPS gate so sparse rows stay sparse. The pool
// is the anchor's nearest localities, precomputed in DuckDB
// (queries.localityPoolByRow); the full vocab when the row has
// no anchor point
private bool injectLocal(ref Components c, Rng* rng, Buckets b,
    const(LocRef)[] pool)
{
    if (b.localities.length == 0 || pool.length == 0
        || rng.unit() >= P_LOCALITY)
        return false;

    auto loc = pool[rng.intIn(0, cast(int) pool.length - 1)];

    c.localities ~= Locality(restyleInitials(rng, loc.name),
        loc.label);

    return true;
}

// sparse rows never reach the grammar
private bool minComps(Components c)
{
    size_t n;

    n += c.door.length > 0;
    n += c.road.length > 0;
    n += c.localities.length;
    n += c.city.length > 0;
    n += c.postcode.length > 0;
    n += c.house.length > 0;
    n += c.unit.length > 0;
    n += c.level.length > 0;

    return n >= MIN_COMPS;
}

// one synthetic example per distinct locality: fake door +
// road + the locality span (the point of the row) + usually
// city and postcode. The surroundings mimic the trade dialect
// so the model learns localities in context
private Components synthComponents(Locality loc, Rng* rng,
    Buckets b, string[][string] pools)
{
    Components c;

    c.door = doorSegs(restyleDoor(
        rng.intIn(1, DOOR_SYNTH_MAX).to!string, rng));

    // real street names when the vocab has them, else the
    // synthetic Main/Cross dialect
    if (b.roads.length > 0 && rng.unit() < P_SYNTH_REAL_ROAD)
    {
        // anchored: a street whose real way geometry passes near
        // the locality — same-name pairs ("<Name> Main Road,
        // <Name>") emerge naturally. The pool is precomputed in
        // DuckDB, keyed by locality name
        string[] pool;

        if (!isNaN(loc.lat) && !isNaN(loc.lon))
            if (auto p = loc.name in pools)
                pool = *p;

        if (pool.length == 0)
            foreach (rd; b.roads)
                pool ~= rd.name;

        c.road = pool[rng.intIn(0, cast(int) pool.length - 1)];
    }
    else
        c.road = synthRoad(rng);

    c.localities ~= loc;

    if (b.cities.length > 0 && rng.unit() < P_SYNTH_CITY)
    {
        c.city = b.cities[rng.intIn(0, cast(int) b.cities.length - 1)];

        dropCity(c.city, rng);
    }

    if (b.postcodes.length > 0 && rng.unit() < P_SYNTH_POSTCODE)
        c.postcode = b.postcodes[rng.intIn(
            0, cast(int) b.postcodes.length - 1)];

    return c;
}

// every distinct spelling of an initial-bearing locality, one
// synth row each: native, dotted, spaced, glued — deduped by
// token stream, deterministic. Plain names stay single
private Locality[] localityVariants(Locality loc)
{
    if (!hasInitialRun(loc.name))
        return [loc];

    Locality[] out_;

    foreach (name; [loc.name, dottedInitials(loc.name),
        normalizeDots(loc.name), gluedInitials(loc.name)])
    {
        bool dup;

        foreach (v; out_)
            if (words(v.name) == words(name))
                dup = true;

        if (!dup)
            out_ ~= Locality(name, loc.label);
    }

    return out_;
}

private size_t synthLocalities(Buckets b,
    string[][string] pools, Rng* rng, File train,
    const int[string] spans)
{
    size_t n;

    foreach (loc; b.localities)
    {
        foreach (v; localityVariants(loc))
        {
            // frequency floor: rare names get extra draws,
            // solid names keep their single row
            auto deficit = LOCALITY_SPAN_FLOOR
                - spans.get(spanKey(words(v.name)), 0);

            if (deficit < 1)
                deficit = 1;

            foreach (i; 0 .. deficit)
            {
                // train-only noise, same as real rows
                auto ex = compose(synthComponents(v, rng, b, pools),
                    rng, b, true);

                if (ex.tokens.length < MIN_TOKENS)
                    continue;

                train.writeln(jsonLine(ex));
                n++;
            }
        }
    }

    return n;
}

// count suburb spans per normalized name — the frequency
// floor's input (train real rows only; the injected
// localities count too, they are real exposure)
private void noteSpans(Example ex, ref int[string] spans)
{
    size_t i;

    while (i < ex.tags.length)
    {
        if (ex.tags[i] != "B-suburb")
        {
            i++;
            continue;
        }

        auto end = i + 1;

        while (end < ex.tags.length
                && ex.tags[end] == "I-suburb")
            end++;

        spans[spanKey(ex.tokens[i .. end])]++;

        i = end;
    }
}

// vocab-comparable span key: word tokens, lowercased —
// dotted and spaced initial spellings share one key
private string spanKey(string[] toks)
{
    string r;

    foreach (t; toks)
    {
        if (t.length == 0 || !isWordTok(t))
            continue;

        if (r.length > 0)
            r ~= ' ';

        r ~= t;
    }

    return r.toLower();
}

// word-char run or multibyte token (punctuation tokens are
// single non-alphanumerics; anything ≥ 0x80 leads a word)
private bool isWordTok(string t)
{
    auto c = t[0];

    if (c >= 0x80)
        return true;

    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z')
        || (c >= 'A' && c <= 'Z');
}

// ---- (locality, street) coverage pairs ----

// every street with a door within ANCHOR_RANGE_M of a
// locality point pairs with it exactly once — coverage, not
// sampling: injectLocal's dice never drew the Kammanahalli
// pair although the pools held it (NOTES 2026-10-11). The
// nearest door wins and rides its city/postcode along;
// locality + city names, lowercased: street values and vocab
// entries matching these are mislabels, not roads (NOTES open 2)
private bool[string] streetDropSet(Buckets b)
{
    bool[string] drop;

    foreach (loc; b.localities)
        drop[loc.name.toLower()] = true;

    foreach (c; b.cities)
        drop[c.toLower()] = true;

    return drop;
}

// ordinal-only streets never pair; per street name only the
// PAIR_MAX_PER_STREET nearest localities survive — the pair
// enumeration itself runs in DuckDB (queries.pairList)

// one composed train-only example per pair; same grammar and
// train-only noise as the synth locality rows
private size_t pairRows(PairRow[] pairs, Buckets b, Rng* rng,
    File train)
{
    size_t n;

    foreach (p; pairs)
    {
        Components c;

        c.road = p.street;
        c.localities ~= Locality(p.locality, p.label);
        c.city = p.city;

        dropCity(c.city, rng);

        c.postcode = p.postcode;

        auto ex = compose(c, rng, b, true);

        if (ex.tokens.length < MIN_TOKENS)
            continue;

        train.writeln(jsonLine(ex));
        n++;
    }

    return n;
}

private int spans(string[] tags, string label)
{
    int n;

    foreach (t; tags)
        n += t == "B-" ~ label;

    return n;
}

private bool isAllDigits(string v)
{
    if (v.length == 0)
        return false;

    foreach (dchar c; v)
        if (!isNumber(c))
            return false;

    return true;
}

private string jsonLine(Example ex)
{
    return `{"tokens": ` ~ jsonList(ex.tokens) ~ `, "tags": `
        ~ jsonList(ex.tags) ~ '}';
}

private string jsonList(string[] items)
{
    import std.array : appender;

    auto buf = appender!(char[])();

    buf ~= '[';

    foreach (i, item; items)
    {
        if (i > 0)
            buf ~= ", ";

        buf ~= jsonStr(item);
    }

    buf ~= ']';

    return buf.data.idup;
}

// json.dumps(ensure_ascii=true) parity: non-ASCII and control
// chars escape as \uXXXX, astral chars as surrogate pairs
private string jsonStr(string s)
{
    import std.array : appender;

    auto buf = appender!(char[])();

    buf ~= '"';

    foreach (dchar c; s)
    {
        switch (c)
        {
        case '"':
            buf ~= "\\\"";
            break;

        case '\\':
            buf ~= "\\\\";
            break;

        case '\n':
            buf ~= "\\n";
            break;

        case '\r':
            buf ~= "\\r";
            break;

        case '\t':
            buf ~= "\\t";
            break;

        case '\b':
            buf ~= "\\b";
            break;

        case '\f':
            buf ~= "\\f";
            break;

        default:
            if (c < 0x20 || c >= 0x7f)
                buf ~= unicodeEscape(c);
            else
                buf ~= c;
        }
    }

    buf ~= '"';

    return buf.data.idup;
}

private string unicodeEscape(dchar c)
{
    import std.format : format;

    auto cp = cast(uint) c;

    if (cp > 0xFFFF)
    {
        auto v = cp - 0x10000;

        return format!"\\u%04x\\u%04x"
            (0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF));
    }

    return format!"\\u%04x"(cp);
}

// rows (osm_id key) to skip, e.g. eval examples
private bool[long] loadExcluded(string path)
{
    bool[long] ids;

    if (path.length == 0)
        return ids;

    foreach (rec; parseJSON(readText(path)).array)
        ids[rec.object["osm_id"].integer] = true;

    return ids;
}

// deterministic test split: md5("osm_type/osm_id") % 100
private int hashPct(string osmType, long osmId)
{
    auto digest = md5Of(osmType ~ "/" ~ osmId.to!string);

    int h;

    foreach (b; digest)
        h = (h * 256 + b) % 100;

    return h;
}

unittest
{
    // deterministic split on osm_type/osm_id; values match the
    // Python builder this tool replaces
    assert(hashPct("node", 42) == 11);
    assert(hashPct("way", 123456) == 40);
    assert(hashPct("node", 9999999999) == 32);

    // json.dumps(ensure_ascii=true): non-ASCII escaped as
    // \uXXXX lowercase, astral chars as surrogate pairs
    assert(jsonStr("ാ") == "\"\\u0d3e\"");
    assert(jsonStr("\U0001F600") == "\"\\ud83d\\ude00\"");
    assert(jsonStr("\b") == "\"\\b\"");
}

unittest
{
    // mislabeled street values blank on real rows too: same-name
    // locality/city values and locality-shaped values (NOTES
    // open 2) — the row survives on its other fields
    Parquet pq;
    pq.texts["door"] = ["12", "12", "12", "12", "12"];
    pq.texts["street"] = ["Adarsh Nagar", "KOZHIKODE",
        "L B S Colony", "PRATAPNAGAR", "MG Road"];
    pq.texts["neighbourhood"] = ["", "", "", "", ""];
    pq.texts["suburb"] = ["", "", "", "", ""];
    pq.texts["quarter"] = ["", "", "", "", ""];
    pq.texts["city"] = ["Bangalore", "Bangalore", "Bangalore",
        "Bangalore", "Bangalore"];
    pq.texts["postcode"] = ["560034", "560034", "560034", "560034",
        "560034"];
    pq.texts["housename"] = ["", "", "", "", ""];
    pq.texts["unit"] = ["", "", "", "", ""];
    pq.texts["level"] = ["", "", "", "", ""];

    auto b = Buckets([], [],
        [Locality("Adarsh Nagar", "suburb")], ["Kozhikode"],
        [], [], [], []);
    auto drop = streetDropSet(b);

    foreach (i; 0 .. 4)
    {
        auto rng = Rng(42);

        assert(rowComponents(pq, i, &rng, false, drop,
            pq.texts["street"][i]).road.length == 0);
    }

    auto rng = Rng(42);

    assert(rowComponents(pq, 4, &rng, false, drop,
        "MG Road").road == "MG Road");
}

unittest
{
    // fill: plain numeric doors re-render to the trade mix (the
    // clean number keeps the house_number span, any
    // decoration sits in front as an O seg); decorated doors
    // pass the restyle through untouched
    Parquet pq;
    pq.texts["door"] = ["12", "G6", ""];
    pq.texts["street"] = ["MG Road", "MG Road", "MG Road"];
    pq.texts["neighbourhood"] = ["", "", ""];
    pq.texts["suburb"] = ["", "", ""];
    pq.texts["quarter"] = ["", "", ""];
    pq.texts["city"] = ["Bangalore", "Bangalore", "Bangalore"];
    pq.texts["postcode"] = ["560034", "560034", "560034"];
    pq.texts["housename"] = ["Shanti Nivas", "", ""];
    pq.texts["unit"] = ["Shop No. 1", "", ""];
    pq.texts["level"] = ["2", "", ""];

    auto rng = Rng(42);

    auto plain = rowComponents(pq, 0, &rng, false, null,
        "MG Road");

    assert(plain.door[$ - 1].label == "house_number");
    assert(plain.door[$ - 1].text.canFind("12"));

    foreach (s; plain.door[0 .. $ - 1])
        assert(s.label == "O");

    auto decorated = rowComponents(pq, 1, &rng, false, null,
        "MG Road");

    assert(decorated.door
        == [Seg("G", "O"), Seg("6", "house_number")]);

    auto none = rowComponents(pq, 2, &rng, false, null, "");

    assert(none.door.length == 0);

    // vocab locality injection: drawn for some seeds, never
    // with an empty vocab; runs after the MIN_COMPS gate so a
    // sparse row is never rescued by an injected locality
    auto b = Buckets([Road("80 Feet Road")], ["Canara Bank"],
        [Locality("Rajendra Nagara", "suburb")],
        ["Bangalore"], ["560034"], [], [], []);

    int injectedCount;

    foreach (seed; 0 .. 32)
    {
        auto srng = Rng(seed);
        auto c = rowComponents(pq, 0, &srng, false, null,
            "MG Road");

        assert(minComps(c));

        if (injectLocal(c, &srng, b,
                [LocRef("Rajendra Nagara", "suburb")]))
        {
            injectedCount++;

            assert(c.localities.length == 1);
            assert(c.localities[0]
                == Locality("Rajendra Nagara", "suburb"));
        }
        else
            assert(c.localities.length == 0);
    }

    assert(injectedCount > 0);

    // droppable fields: train-only dropout, never without noise;
    // addr:neighbourhood maps to the merged suburb label
    pq.texts["neighbourhood"] = ["Jayanagar", "", ""];

    int dropped;

    foreach (seed; 0 .. 32)
    {
        auto srng = Rng(seed);

        auto quiet = rowComponents(pq, 0, &srng, false, null,
            "MG Road");

        assert(quiet.localities.length == 1);
        assert(quiet.localities[0]
            == Locality("Jayanagar", "suburb"));

        auto noisy = rowComponents(pq, 0, &srng, true, null,
            "MG Road");

        dropped += noisy.localities.length == 0;
    }

    assert(dropped > 0);

    // sparse rows: one component is below MIN_COMPS, before
    // and regardless of injection
    auto sparse = Components(Components.init.door, "MG Road",
        [], "", "", "", "", "");

    assert(!minComps(sparse));

    // housename, unit and level columns fill their components
    // and count towards MIN_COMPS
    auto h = rowComponents(pq, 0, &rng, false, null, "MG Road");

    assert(h.house == "Shanti Nivas");
    assert(h.unit == "Shop No. 1");
    assert(h.level == "2");
    assert(minComps(h));
}

unittest
{
    // street dropout: train-only, the same noise family as the
    // droppable locality fields — the model must learn to emit
    // no road at all (roadless trade and hand-eval addresses);
    // never without noise, absent streets stay absent
    Parquet pq;
    pq.texts["door"] = ["12"];
    pq.texts["street"] = ["MG Road"];
    pq.texts["neighbourhood"] = [""];
    pq.texts["suburb"] = [""];
    pq.texts["quarter"] = [""];
    pq.texts["city"] = ["Bangalore"];
    pq.texts["postcode"] = ["560034"];
    pq.texts["housename"] = [""];
    pq.texts["unit"] = [""];
    pq.texts["level"] = [""];

    int dropped;

    foreach (seed; 0 .. 256)
    {
        auto srng = Rng(seed);

        assert(rowComponents(pq, 0, &srng, false, null,
            "MG Road").road == "MG Road");

        auto c = rowComponents(pq, 0, &srng, true, null,
            "MG Road");

        assert(c.road == "MG Road" || c.road.length == 0);
        dropped += c.road.length == 0;
    }

    assert(dropped > 0);

    pq.texts["street"] = [""];

    foreach (seed; 0 .. 64)
    {
        auto srng = Rng(seed);

        assert(rowComponents(pq, 0, &srng, true, null, "").road.length == 0);
    }
}

unittest
{
    // city dropout: trade input carries no city column — the
    // last locality then sits in the slot before the PIN and
    // the model city-tags it (impute hand50: RAJAJINAGAR,
    // Mallathahalli, BILEKAHALLI). Train-only, never without
    // noise, absent cities stay absent
    Parquet pq;
    pq.texts["door"] = ["12"];
    pq.texts["street"] = ["MG Road"];
    pq.texts["neighbourhood"] = [""];
    pq.texts["suburb"] = ["Jayanagar"];
    pq.texts["quarter"] = [""];
    pq.texts["city"] = ["Bangalore"];
    pq.texts["postcode"] = ["560034"];
    pq.texts["housename"] = [""];
    pq.texts["unit"] = [""];
    pq.texts["level"] = [""];

    int dropped;

    foreach (seed; 0 .. 256)
    {
        auto srng = Rng(seed);

        assert(rowComponents(pq, 0, &srng, false, null,
            "MG Road").city == "Bangalore");

        auto c = rowComponents(pq, 0, &srng, true, null,
            "MG Road");

        assert(c.city == "Bangalore" || c.city.length == 0);
        dropped += c.city.length == 0;
    }

    assert(dropped > 0);

    pq.texts["city"] = [""];

    foreach (seed; 0 .. 64)
    {
        auto srng = Rng(seed);

        assert(rowComponents(pq, 0, &srng, true, null,
            "").city.length == 0);
    }
}

unittest
{
    // synthetic locality example: fake door, a road, and
    // exactly one locality span carrying the right label
    auto b = Buckets([Road("MG Road")], ["Canara Bank"],
        [Locality("Koramangala", "suburb")],
        ["Bangalore"], ["560034"], [], [], []);
    auto rng = Rng(42);

    auto ex = compose(
        synthComponents(Locality("Koramangala", "suburb"), &rng, b,
            null),
        &rng, b, true);

    assert(ex.tokens.length == ex.tags.length);
    assert(ex.tokens.length >= 3);

    // the locality span is present exactly once
    int bSuburb;

    foreach (t; ex.tags)
        bSuburb += t == "B-suburb";

    assert(bSuburb == 1);

    // door and road spans exist too
    bool door, road;

    foreach (t; ex.tags)
    {
        door = door || t == "B-house_number";
        road = road || t == "B-road";
    }

    assert(door && road);
}

unittest
{
    // city dropout on the synth path: train-only rows lose
    // the city at the same rate as real rows — the compound
    // area + locality frame must render cityless too
    auto b = Buckets([Road("MG Road")], [],
        [Locality("Koramangala", "suburb")], ["Bangalore"],
        ["560034"], [], [], []);

    int dropped;

    foreach (seed; 0 .. 512)
    {
        auto rng = Rng(seed);

        auto c = synthComponents(
            Locality("Koramangala", "suburb"), &rng, b, null);

        assert(c.city == "Bangalore" || c.city.length == 0);
        dropped += c.city.length == 0;
    }

    // P_NO_CITY stacks on the P_SYNTH_CITY baseline (15%
    // cityless) up to ≈32% — a bare > 0 would pass without
    // the dropout
    assert(dropped > 512 * 0.25);
}

unittest
{
    // locality frequency floor (open 2 / Fix D): the median
    // vocab locality renders exactly once (its synth row) —
    // the decision margin that absorbed GOTTIGERE into
    // road. Names the real rows rendered fewer than
    // LOCALITY_SPAN_FLOOR times get extra synth draws;
    // solid names keep their single row. Dotted and spaced
    // spellings share one key
    import std.file : tempDir, remove;

    auto b = Buckets([Road("MG Road")], [],
        [Locality("Kothnur", "suburb"),
            Locality("Gottigere", "suburb")],
        ["Bangalore"], ["560034"], [], [], []);

    // Kothnur solid, Gottigere never rendered by a real row
    int[string] spanCounts;
    spanCounts[spanKey(["Kothnur"])] = LOCALITY_SPAN_FLOOR;

    auto path = tempDir ~ "/synth_floor_test.jsonl";

    {
        auto tmp = File(path, "w");
        auto rng = Rng(42);

        assert(synthLocalities(b, null, &rng, tmp, spanCounts)
            == LOCALITY_SPAN_FLOOR + 1);
    }

    int gottigere;

    foreach (line; File(path).byLine)
        gottigere += line.toLower().canFind("gottigere");

    remove(path);

    assert(gottigere == LOCALITY_SPAN_FLOOR);

    // spellings share one key
    assert(spanKey(["J", ".", "C", ".", "Nagar"])
        == spanKey(["J", "C", "Nagar"]));
    assert(spanKey(["Gottigere"]) == "gottigere");
}

unittest
{
    // every initial-bearing locality renders each distinct
    // spelling as its own synth row — deterministic, deduped
    // by token stream; plain names stay single
    auto variants = localityVariants(Locality("J C Nagar", "suburb"));

    assert(variants == [
        Locality("J C Nagar", "suburb"),
        Locality("J.C.Nagar", "suburb"),
        Locality("JC Nagar", "suburb"),
    ]);

    // natively dotted: all four spellings are distinct token
    // streams — "A.C Nagar" dots only the initial run,
    // "A.C.Nagar" dots the head too
    auto dottedVariants = localityVariants(
        Locality("A.C Nagar", "suburb"));

    assert(dottedVariants == [
            Locality("A.C Nagar", "suburb"),
            Locality("A.C.Nagar", "suburb"),
            Locality("A C Nagar", "suburb"),
            Locality("AC Nagar", "suburb"),
        ], format("%s", dottedVariants));

    // single initial: glued duplicates spaced, native and
    // dotted share one token stream
    assert(localityVariants(Locality("A.Gokulapadu", "suburb"))
        == [
            Locality("A.Gokulapadu", "suburb"),
            Locality("A Gokulapadu", "suburb"),
        ]);

    assert(localityVariants(Locality("Koramangala", "suburb"))
        == [Locality("Koramangala", "suburb")]);
}

unittest
{
    // injection draws from the precomputed SQL pool only; the
    // range/cap/fallback semantics live in
    // queries.localityPoolByRow
    Locality[] locs;

    foreach (i; 0 .. 60)
        locs ~= Locality("L" ~ i.to!string, "suburb",
            12.90 + i * 0.01, 77.60);

    auto b = Buckets([Road("MG Road")], [],
        locs, [], [], [], [], []);

    int drawn;

    foreach (seed; 0 .. 200)
    {
        auto srng = Rng(seed);
        auto c = Components.init;
        c.road = "MG Road";

        if (!injectLocal(c, &srng, b,
                [LocRef("L5", "suburb"), LocRef("L6", "suburb")]))
            continue;

        drawn++;

        assert(c.localities.length == 1);

        auto i = to!int(c.localities[0].name[1 .. $]);

        assert(i == 5 || i == 6, format("off-pool locality drawn: %s", i));
    }

    assert(drawn > 0);

    // empty pool never injects (the range/cap/fallback
    // semantics live in queries.localityPoolByRow)
    foreach (seed; 0 .. 32)
    {
        auto srng = Rng(seed);
        auto c = Components.init;

        assert(!injectLocal(c, &srng, b, []));
    }
}

unittest
{
    // synthetic rows anchor the road to the locality via the
    // precomputed pool (queries.roadPoolByLocality): with 40
    // roads 1.1 km apart, the 800 m pool is the locality's own
    // road (index 5). Point-less locality → uniform pool
    Road[] roads;

    foreach (i; 0 .. 40)
        roads ~= Road("R" ~ i.to!string);

    auto b = Buckets(roads, [],
        [Locality("Koramangala", "suburb", 12.95, 77.60)],
        [], [], [], [], []);

    string[][string] pools;
    pools["Koramangala"] = ["R5"];

    int drawn, far;

    foreach (seed; 0 .. 200)
    {
        auto rng = Rng(seed);

        auto c = synthComponents(
            Locality("Koramangala", "suburb", 12.95, 77.60),
            &rng, b, pools);

        // synthRoad draws ("3rd Cross") start with a digit
        if (c.road.length == 0 || c.road[0] != 'R')
            continue;

        drawn++;

        auto i = to!int(c.road[1 .. $]);

        assert(i <= 24, format("far road drawn: %s", i));
    }

    assert(drawn > 0);

    foreach (seed; 0 .. 400)
    {
        auto rng = Rng(seed);

        auto c = synthComponents(
            Locality("Koramangala", "suburb"), &rng, b, pools);

        if (c.road.length == 0 || c.road[0] != 'R')
            continue;

        far += to!int(c.road[1 .. $]) > 24;
    }

    assert(far > 0);
}

unittest
{
    // rows: one composed train-only example per (locality,
    // street) pair from queries.pairList; the locality span
    // carries the bucket label
    auto b = Buckets([], [],
        [Locality("Kammanahalli", "suburb")],
        [], [], [], [], []);

    PairRow[] pairs = [PairRow("Kammanahalli", "suburb",
        "Kammanahalli Main Road", "Mysuru", "570001")];

    import std.file : tempDir, remove;
    auto path = tempDir ~ "/pair_rows_test.jsonl";
    {
        auto tmp = File(path, "w");
        auto rng = Rng(42);
        assert(pairRows(pairs, b, &rng, tmp) == 1);
    }

    int rows;

    foreach (line; File(path).byLine)
    {
        rows++;

        assert(line.canFind("B-road"));
        assert(line.canFind("B-suburb"));
    }

    remove(path);

    assert(rows == 1);
}

// ---- block→parent pairs ----

// one block + its parent settlement ("4th Block", "Jayanagar")
private struct BlockPair
{
    string block;
    string parent;
    double lat;
    double lon;
}

// block-named areas ("<…> Block"); revenue blocks
// ("Mundakkayam Block") ride along, they are real places
private bool isBlockArea(string name)
{
    auto toks = name.split;

    return toks.length >= 2 && toks[$ - 1].toLower() == "block";
}

// ward polygons named by number only ("Ward 100") teach
// nothing — trade writes the settlement name
private bool isWardNumber(string name)
{
    auto toks = name.split;

    return toks.length == 2 && toks[0].toLower() == "ward"
        && isAllDigits(toks[1]);
}

// settlement-scale parents only: local admin 9/10 and place
// polygons; districts and taluks stay out
private bool isParentArea(string name, string label,
    string adminLevel)
{
    if (name.length == 0 || isBlockArea(name)
        || isWardNumber(name))
        return false;

    return adminLevel == "9" || adminLevel == "10"
        || label.length > 0;
}

// smallest containing parent by polygon area; every block never
// parents — hits named like the block cover the block inside a
// same-named block
private const(AreaHit)* pickParent(const AreaHit[] hits,
    string block)
{
    const(AreaHit)* best;

    foreach (ref h; hits)
    {
        if (h.name.toLower() == block.toLower())
            continue;

        if (!isParentArea(h.name, h.label, h.adminLevel))
            continue;

        if (best is null || h.m2 < best.m2)
            best = &h;
    }

    return best;
}

// "Jayanagar 4th Block" inside "Jayanagar" → "4th Block"; no
// parent prefix → the name stays whole
private string bareBlockName(string block, string parent)
{
    auto p = parent ~ " ";

    if (block.length > p.length
        && block[0 .. p.length].toLower() == p.toLower())
        return block[p.length .. $];

    return block;
}

private void addPair(ref BlockPair[] pairs,
    ref bool[string] seen, string block, string parent,
    double lat, double lon)
{
    auto key = block.toLower() ~ '\0' ~ parent.toLower();

    if (key in seen)
        return;

    seen[key] = true;
    pairs ~= BlockPair(block, parent, lat, lon);
}

// candidates pair via their point — one row per distinct block
// name, deduped by stage 1 (areas first: the polygon back is the
// reliable anchor). All containment queries batch into one SQL
// call (queries.containing); old bucket dirs without the file
// emit nothing
private BlockPair[] blockList(string dir)
{
    auto path = dir ~ "/block_candidates.parquet";

    if (!exists(path))
        return null;

    auto names = loadText(path, ["name"])["name"];
    auto pts = loadNums(path, ["lat", "lon"]);

    double[][] points;

    foreach (i, name; names)
        points ~= [pts["lat"][i], pts["lon"][i]];

    auto hits = containing(dir, points);

    BlockPair[] pairs;
    bool[string] seen;

    foreach (i, name; names)
    {
        auto parent = pickParent(hits[i], name);

        if (parent is null)
            continue;

        addPair(pairs, seen, bareBlockName(name, parent.name),
            parent.name, points[i][0], points[i][1]);
    }

    return pairs;
}

// one composed train-only example per pair, synth-shaped
// around the block; the parent rides as the second suburb
// span after it ("4th Block, Jayanagar")
private size_t blockRows(BlockPair[] pairs, Buckets b,
    string[][string] pools, Rng* rng, File train)
{
    size_t n;

    foreach (p; pairs)
    {
        auto c = synthComponents(
            Locality(p.block, "suburb", p.lat, p.lon), rng, b, pools);

        c.localities ~= Locality(p.parent, "suburb");

        auto ex = compose(c, rng, b, true);

        if (ex.tokens.length < MIN_TOKENS)
            continue;

        train.writeln(jsonLine(ex));
        n++;
    }

    return n;
}

unittest
{
    // candidates pair with the smallest containing named area —
    // the trade street field writes the block and its parent
    // settlement as adjacent suburb spans ("4th Block,
    // Jayanagar"). The parent prefix strips from the block name;
    // parents named by ward number only, other blocks, nameless
    // and non-settlement admin (taluk) never parent.
    // The fixture block_candidates.parquet mirrors stage 1's
    // output: one row per distinct block name, ORDER BY name
    import std.array : join;

    import address_generator.parquet_input : exec;
    import bucket_generator.wkb : wkbMultiPolygon, WRing = Ring;

    // the parent polygons (areas.geoparquet): Jayanagar ward
    // around its 4th block; a "Ward 100" polygon and a taluk sit
    // between — both must lose the smallest contest by the
    // parent gates, not by size. (WRing(lon[], lat[]))
    string areaRow(long id, string name, string label,
        string adminLevel, double m2, WRing r)
    {
        auto lon = r.lon.dup ~ r.lon[0];
        auto lat = r.lat.dup ~ r.lat[0];
        auto w = wkbMultiPolygon([WRing(lon, lat)], []);

        return format("SELECT 'R' AS osm_type, %s AS osm_id,"
            ~ " '%s' AS name, '%s' AS label, '%s' AS admin_level,"
            ~ " %s AS area, ST_GeomFromWKB(unhex('%(%02x%)'))"
            ~ " AS geometry",
            id.to!string, name, label, adminLevel,
            format("%.17g", m2), w);
    }

    enum dir = "/tmp/block_fixture";

    import std.file : mkdirRecurse;
    mkdirRecurse(dir);

    string[] polyRows = [
        areaRow(1, "Jayanagar", "", "10", 400.0,
            WRing([77.58, 77.62, 77.62, 77.58],
                  [12.90, 12.90, 12.92, 12.92])),
        areaRow(2, "Ward 100", "", "10", 200.0,
            WRing([77.582, 77.602, 77.602, 77.582],
                  [12.902, 12.902, 12.912, 12.912])),
        areaRow(3, "Anekal Taluk", "", "6", 800.0,
            WRing([77.40, 77.80, 77.80, 77.40],
                  [12.80, 12.80, 13.00, 13.00])),
        // quarter-tagged block polygons: the block gate, not
        // the label or size, keeps them from parenting
        areaRow(4, "Jayanagar 4th Block", "quarter", "", 50.0,
            WRing([77.585, 77.60, 77.60, 77.585],
                  [12.905, 12.905, 12.91, 12.91])),
        areaRow(5, "HSR Layout", "suburb", "", 300.0,
            WRing([77.50, 77.55, 77.55, 77.50],
                  [12.90, 12.90, 12.91, 12.91])),
        areaRow(6, "4th Block", "neighbourhood", "", 10.0,
            WRing([77.51, 77.54, 77.54, 77.51],
                  [12.902, 12.902, 12.908, 12.908])),
        // nameless containment never parents
        areaRow(7, "", "", "10", 20.0,
            WRing([77.52, 77.53, 77.53, 77.52],
                  [12.902, 12.902, 12.908, 12.908])),
        areaRow(8, "Kommagatta", "", "9", 600.0,
            WRing([76.10, 76.20, 76.20, 76.10],
                  [13.10, 13.10, 13.12, 13.12])),
    ];

    exec("COPY (" ~ polyRows.join(" UNION ALL ")
        ~ ") TO '" ~ dir ~ "/areas.geoparquet' (FORMAT parquet)");

    // one row per block name: two polygon-backed areas, three
    // locality points. "Jayanagar 4th Block" strips and pairs
    // ("4th Block", "Jayanagar") — a different parent than the
    // bare area's, so both survive; "Jayanagar Block" strips
    // onto the bare "Block" pair and dedupes away
    exec("COPY (SELECT * FROM (VALUES"
        ~ " ('4th Block', 12.905, 77.525),"
        ~ " ('7th Block', 13.11, 76.15),"
        ~ " ('9th Block', 12.85, 77.45),"
        ~ " ('Block', 12.9065, 77.5955),"
        ~ " ('Jayanagar 4th Block', 12.9075, 77.5925),"
        ~ " ('Jayanagar Block', 12.9075, 77.5925)"
        ~ ") t(name, lat, lon)) TO '" ~ dir
        ~ "/block_candidates.parquet' (FORMAT parquet)");

    auto pairs = blockList(dir);

    assert(pairs.length == 4, format("%s", pairs));

    // bare block area: its own polygon (same name) never parents,
    // the nameless and taluk hits lose the gates, HSR Layout wins
    assert(pairs[0].block == "4th Block");
    assert(pairs[0].parent == "HSR Layout");
    assert(abs(pairs[0].lat - 12.905) < 1e-9);
    assert(abs(pairs[0].lon - 77.525) < 1e-9);

    // locality point: the zone parented, the taluk-only
    // locality dropped
    assert(pairs[1] == BlockPair("7th Block", "Kommagatta",
        13.11, 76.15));

    // area points: parent prefix stripped, smallest valid parent
    // won (Jayanagar over Ward 100 and the taluk)
    assert(pairs[2] == BlockPair("Block", "Jayanagar",
        12.9065, 77.5955));
    assert(pairs[3] == BlockPair("4th Block", "Jayanagar",
        12.9075, 77.5925));

    // rows: one train-only synth-shaped example per pair, the
    // block and parent render as two suburb spans
    import std.file : tempDir, remove;

    auto b = Buckets([Road("MG Road")], [],
        [Locality("Koramangala", "suburb")],
        ["Bangalore"], ["560034"], [], [], []);

    auto path = tempDir ~ "/block_rows_test.jsonl";

    {
        auto tmp = File(path, "w");
        auto rng = Rng(42);

        assert(blockRows(pairs, b, null, &rng, tmp) == 4);
    }

    int rows;

    foreach (line; File(path).byLine)
    {
        rows++;

        // two B-suburb spans: block and parent
        assert(line.canFind("B-suburb"));
        assert(line.split("B-suburb").length >= 3,
            format("row: %s", line));
    }

    remove(path);

    assert(rows == 4);
}
