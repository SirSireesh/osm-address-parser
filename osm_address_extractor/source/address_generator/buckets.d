// address_generator bucket loader: the unit vocabularies extracted by
// bucket_generator (bucket_generator.vocab), read back from the bucket dir:
//
//   <dir>/streets.parquet    one line per named road way; the
//                              street vocab is DISTINCT name over it
//   <dir>/pois.parquet         landmark POI names
//   <dir>/localities.parquet   name + suburb/quarter/neighbourhood
//                              label + first-seen point
//   <dir>/cities.parquet      city names, frequency-ranked
//   <dir>/postcodes.parquet    6-digit PINs
//
//   addresses.parquet is loaded by dataset.d via
//   parquet_input.load (it carries osm_type/osm_id for the
//   deterministic split).
module address_generator.buckets;

import std.file : exists;

import address_generator.coords : samePoint;
import address_generator.gates : isLocalityStreet;
import address_generator.parquet_input : loadNums, loadText;

// one settlement name with its BIO label and first-seen point
// (NaN = none)
public struct Locality
{
    string name;
    string label;
    double lat = double.nan;
    double lon = double.nan;

    // NaN means "no point": missing points compare equal
    public bool opEquals(const Locality o) const
    {
        return name == o.name && label == o.label
            && samePoint(lat, o.lat) && samePoint(lon, o.lon);
    }
}

// one street vocab name (geometry lives in streets.parquet)
public struct Road
{
    string name;

    public bool opEquals(const Road o) const
    {
        return name == o.name;
    }
}

// the unit buckets, ready for the grammar
public struct Buckets
{
    Road[] roads;
    string[] pois;
    Locality[] localities;
    string[] cities;
    string[] postcodes;

    // premises detail
    string[] housenames;
    string[] units;
    string[] levels;
}

public Buckets loadBuckets(string dir)
{
    Buckets b;

    b.pois = loadText(dir ~ "/pois.parquet", ["name"])["name"];

    auto locs = loadText(dir ~ "/localities.parquet",
        ["name", "label"]);
    auto locPts = loadNums(dir ~ "/localities.parquet",
        ["lat", "lon"]);

    foreach (i, name; locs["name"])
        b.localities ~= Locality(name, locs["label"][i],
            locPts["lat"][i], locPts["lon"][i]);

    b.cities = loadText(dir ~ "/cities.parquet", ["name"])["name"];
    b.postcodes = loadText(dir ~ "/postcodes.parquet",
        ["name"])["name"];

    // bucket dirs written before these files existed have no
    // premises buckets: treat missing as empty
    b.housenames = loadOptional(dir ~ "/housenames.parquet");
    b.levels = loadOptional(dir ~ "/levels.parquet");

    // the unit/street vocab anti-joins run as SQL over the bucket
    // files (queries.d); the locality-shape street predicate stays
    // here — a pure string test, not a data join
    import address_generator.queries : unitNames, roadNames;

    b.units = exists(dir ~ "/units.parquet")
        ? unitNames(dir) : [];

    foreach (nm; roadNames(dir))
        if (!isLocalityStreet(nm))
            b.roads ~= Road(nm);

    return b;
}

private string[] loadOptional(string path)
{
    return exists(path)
        ? loadText(path, ["name"])["name"] : [];
}
