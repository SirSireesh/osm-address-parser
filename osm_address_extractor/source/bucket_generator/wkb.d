// WKB encoding for the areas side-output: place and local
// admin boundary polygons persist as GeoParquet geometry —
// address_generator containment queries (block→parent suburb, door→
// ward) need shapes, not centroids.
//
//   wkbMultiPolygon([outer], [holes])
//     → 01 06000000 <nPolygons>
//       01 03000000 <nRings> <ring>...
//       ring = <nPoints> (lon,lat) pairs, little-endian doubles
module bucket_generator.wkb;

import std.algorithm.iteration : filter;
import std.array : array;

// little-endian WKB geometry type codes
private enum WKB_POLYGON = 3;
private enum WKB_MULTIPOLYGON = 6;

// one closed ring: x = lon (OGC:CRS84), y = lat
public struct Ring
{
    double[] lon;
    double[] lat;
}

// MultiPolygon of the outer rings, each hole assigned to the
// outer whose bbox contains it (boundary relations nest
// directly); unassigned holes drop. Null when no outer ring
public ubyte[] wkbMultiPolygon(const Ring[] outers,
    const Ring[] inners)
{
    if (outers.length == 0)
        return null;

    ubyte[] buf;

    putHeader(buf, WKB_MULTIPOLYGON, cast(uint) outers.length);

    foreach (o; outers)
    {
        // this hole's bbox fits inside this outer's bbox
        auto holes = inners.filter!(
            h => h.lon.length >= 3 && bboxHas(o, h)).array;

        putHeader(buf, WKB_POLYGON, cast(uint) holes.length + 1);

        putRing(buf, o);

        foreach (h; holes)
            putRing(buf, h);
    }

    return buf;
}

private void putHeader(ref ubyte[] buf, uint type, uint count)
{
    import std.bitmanip : nativeToLittleEndian;

    buf ~= cast(ubyte) 1;                        // little-endian
    buf ~= nativeToLittleEndian(type)[];
    buf ~= nativeToLittleEndian(count)[];
}

// ring = point count, then (lon, lat) little-endian pairs
private void putRing(ref ubyte[] buf, const Ring r)
{
    import std.bitmanip : nativeToLittleEndian;

    buf ~= nativeToLittleEndian(cast(uint) r.lon.length)[];

    foreach (i; 0 .. r.lon.length)
    {
        buf ~= nativeToLittleEndian(*cast(ulong*) &r.lon[i])[];
        buf ~= nativeToLittleEndian(*cast(ulong*) &r.lat[i])[];
    }
}

// the inner bbox sits inside the outer bbox
private bool bboxHas(const Ring outer, const Ring inner)
{
    import std.algorithm.comparison : max, min;

    double oLo = outer.lon[0], oHi = oLo, iLo = inner.lon[0],
        iHi = iLo;
    double oLa = outer.lat[0], oHa = oLa, iLa = inner.lat[0],
        iHa = iLa;

    foreach (i; 1 .. outer.lon.length)
    {
        oLo = min(oLo, outer.lon[i]);
        oHi = max(oHi, outer.lon[i]);
        oLa = min(oLa, outer.lat[i]);
        oHa = max(oHa, outer.lat[i]);
    }

    foreach (i; 1 .. inner.lon.length)
    {
        iLo = min(iLo, inner.lon[i]);
        iHi = max(iHi, inner.lon[i]);
        iLa = min(iLa, inner.lat[i]);
        iHa = max(iHa, inner.lat[i]);
    }

    return iLo >= oLo && iHi <= oHi && iLa >= oLa && iHa <= oHa;
}

unittest
{
    // independent decode: a square outer with a triangular
    // hole round-trips through the byte layout
    auto outer = Ring([0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0]);
    auto hole = Ring([0.2, 0.2, 0.4], [0.2, 0.2, 0.4]);

    auto b = wkbMultiPolygon([outer], [hole]);

    assert(b !is null);
    assert(b.length == 9 + 9 + 4 + 4 * 16 + 4 + 3 * 16);

    assert(b[0] == 1);                              // little-endian
    assert(b[1 .. 5] == cast(ubyte[]) [6, 0, 0, 0]); // MultiPolygon
    assert(b[5 .. 9] == cast(ubyte[]) [1, 0, 0, 0]); // 1 polygon

    size_t at = 9;

    assert(b[at] == 1);                             // little-endian
    assert(b[at + 1 .. at + 5]
        == cast(ubyte[]) [3, 0, 0, 0]);             // Polygon
    assert(b[at + 5 .. at + 9]
        == cast(ubyte[]) [2, 0, 0, 0]);             // outer + hole

    at += 9;

    assert(b[at .. at + 4] == cast(ubyte[]) [4, 0, 0, 0]);
    at += 4;

    double leDouble(const(ubyte)[] x)
    {
        ulong v;

        foreach_reverse (ubyte c; x)
            v = (v << 8) | c;

        return *cast(double*) &v;
    }

    // first point of the outer ring: lon then lat
    assert(leDouble(b[at .. at + 8]) == 0.0);
    assert(leDouble(b[at + 8 .. at + 16]) == 0.0);

    at += 4 * 16;

    assert(b[at .. at + 4] == cast(ubyte[]) [3, 0, 0, 0]);
    at += 4;

    assert(leDouble(b[at .. at + 8]) == 0.2);
    assert(leDouble(b[at + 8 .. at + 16]) == 0.2);

    // a hole whose bbox escapes the outer is not assigned
    auto far = Ring([5.0, 5.0, 5.4], [5.0, 5.0, 5.4]);

    auto c = wkbMultiPolygon([outer], [hole, far]);

    assert(c[14 .. 18] == cast(ubyte[]) [2, 0, 0, 0]); // one hole kept

    // no outers → null
    assert(wkbMultiPolygon([], []) is null);
}
