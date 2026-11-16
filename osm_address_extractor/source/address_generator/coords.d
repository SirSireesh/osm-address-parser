// Shared coordinate constants and bucket-point equality; all
// distance math lives in DuckDB (queries.d)
module address_generator.coords;

import std.math : isNaN;

// meters per degree of latitude; longitude degrees shrink by
// cos(lat)
public enum double DEG_TO_M = 111_320;

// NaN means "no point", not a value: two missing points
// compare equal, so bucket entries without coordinates stay
// testable (struct opEquals)
public bool samePoint(double a, double b)
{
    return (isNaN(a) && isNaN(b)) || a == b;
}

