// The address grammar: components → complete tagged addresses.
//
// bucket_generator extracts clean units into buckets; this module
// composes them into BIO-tagged strings, libpostal's
// AddressFormatter role:
//
//   door   road   road+   [landmark]   localities   city   postcode
//     │     │      │         │             │          │       │
//     ▼     ▼      ▼         ▼             ▼          ▼       ▼
//   "12/7, 80 Feet Road Rajendra Nagara, Bangalore, 560034"
//
// Slot order mirrors the trade dialect: a second road follows
// the first, the landmark sits after the roads, glued to the
// road, right after the door, in the locality slot, or at the
// very end (30%), localities precede the city — the first may
// glue to the road comp without a comma ("80 Feet Road
// Rajendra Nagara") or sit in front of a road ("Gear School
// Road, Panathur, Outer Ring Road"), the rest get their own
// comps.
// Premises detail (unit, level, housename) sits between the
// door and the road; compounds (10%) add a second housename
// (the building) and an area right before the road.
//
// compose() renders a template string —
// "{door}, {road}, {road}, {locality}, {city}, {postcode}":
// "{slot}" holes pull segs from the row components / buckets,
// a repeated slot consumes successive values ("{road}, {road}"
// is a multi-road street, no per-slot code), "," separates
// comps, "~" glues the next slot into the previous comp
// (comma fallback under MIN_ROW_TOKENS), "+" glues
// unconditionally, any other literal text (" - ") renders O.
module address_generator.grammar;

import std.algorithm.searching : canFind, countUntil;
import std.array : join, split;
import std.ascii : isDigit;
import std.conv : to;
import std.format : format;
import std.random : Mt19937, uniform;
import std.string : toLower, toUpper;
import std.uni : isAlpha, isMark, isNumber, isWhite, toLower,
    toUpper;

import address_generator.buckets : Buckets, Locality, Road;
import address_generator.templates : pickSkeleton;

// second-road and landmark vocab-vs-synth shares
private enum P_REAL_SECOND_ROAD = 0.50;
private enum P_REAL_NAME_LANDMARK = 0.70;

// residential compounds: "<building>, <area>" sits between
// the premises detail and the road ("G4, Tower
// C, Divyasree Republic of Whitefield, Kundalahalli, ...").
// The area is a vocab locality (named landuse areas live
// there — residential, industrial, commercial alike), the
// building an addr:housename or a synthesized style
private enum P_REAL_BUILDING = 0.50;

private enum P_UNIT = 0.05;
private enum P_LEVEL = 0.08;
private enum P_HOUSE = 0.05;
private enum P_REAL_UNIT = 0.70;

private enum P_UNIT_PAIR = 0.15;
private enum P_UNIT_BARE = 0.15;
private enum P_UNIT_KIND_HYPHEN = 0.25;

// floors ("2nd Floor") and synthesized unit styles
private enum string[] LEVELS = [
    "Ground Floor", "1st Floor", "2nd Floor", "3rd Floor",
    "4th Floor", "Basement",
];
private enum string[] UNIT_KINDS = ["Shop No.", "Unit No.",
    "Flat", "Suite", "Room"];
private enum UNIT_SUB_MAX = 300;

// redraw budget before the compound is dropped: the area must
// not duplicate a locality the row already carries
private enum AREA_DRAWS = 3;

// synthesized building names: "Tower C", "Block B", "Wing D"
private enum string[] BUILDING_KINDS = ["Tower", "Block", "Wing"];

// trade writers omit the comma between fields at random
// (measured: locality→city 37.5%, road→level 44.9%,
// road→road 40.3% commaless; full rows run comma-free 2x
// the independent rate — a writer style, not per-address
// mixing). Row style + per-boundary dropout lands both
// marginals at ≈0.40
private enum P_ROW_COMMALESS = 0.25;
private enum P_BOUNDARY_COMMALESS = 0.20;

// dataset.d drops examples below MIN_TOKENS = 3 — the
// commaless rendering must not push rows under that floor
private enum MIN_ROW_TOKENS = 3;

// train-only noise: lowercase the whole row. No comp
// shuffling: slot order is the grammar's contract, a
// permuted row teaches an impossible address
private enum P_LOWERCASE = 0.10;

// case mix for landmark and locality spans: trade fields
// render ALL CAPS, OSM is mostly Title Case. The label must
// attach to the marker and the position, not to a casing
// heuristic ("KAMMANAHALLI" was tagged road for being caps)
private enum P_CASE_CAPS = 0.45;
private enum P_CASE_LOWER = 0.10;

// marker distribution from trade data:
// NEAR 3.1%, OPP. 0.6%, BEHIND 0.5%, NEXT TO 0.4%
private struct Marker
{
    string word;
    int weight;
}

private enum Marker[] MARKERS = [
    Marker("NEAR", 31),
    Marker("OPP.", 6),
    Marker("BEHIND", 5),
    Marker("NEXT TO", 4),
];

// generic landmarks otherwise: the most frequent category names
// in the southern-zone OSM POI vocab (Anganwadi 2501, Ration
// Shop 1633, State Bank of India 712, School Ground 423,
// Government Hospital 309, Canara Bank 284, Indian Oil 199,
// Primary Health Centre 146, Bus Stop 98, Post Office 50) —
// frequency IS genericness here; no trade-license input
private enum string[] GENERIC_LANDMARKS = [
    "Anganwadi", "Ration Shop", "State Bank of India",
    "School Ground", "Government Hospital", "Canara Bank",
    "Indian Oil", "Primary Health Centre", "Bus Stop",
    "Post Office",
];

private enum ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

private enum ORD_MAX = 30;
private enum P_ROAD_SUFFIX = 0.30;

private enum PLOT_SUB_MAX = 999;

// plain-numeric OSM doors re-rendered to the BBMP trade mix.
// Weights are calibrated on southern-zone: 84% of surviving
// doors are plain (junk filtered out, plot/alpha/no/hash
// formats pass through), so these shares of the plain bucket
// land the output on the trade distribution (plain 38.6%,
// plot 23.7%, No-prefixed 10.8%, # 7.4%, alpha 11.2%,
// ward-code 3.9%)
private enum DoorStyle { plain, plot, no, hash, alpha, ward, syno }

private struct DoorWeight
{
    DoorStyle style;
    int weight;
}

private enum DoorWeight[] DOOR_WEIGHTS = [
    DoorWeight(DoorStyle.plain, 48),
    DoorWeight(DoorStyle.plot, 15),
    DoorWeight(DoorStyle.no, 12),
    DoorWeight(DoorStyle.hash, 7),
    DoorWeight(DoorStyle.alpha, 12),
    DoorWeight(DoorStyle.ward, 5),
    // trade survey doors: 0.8% in trade, ~0.2% in OSM
    DoorWeight(DoorStyle.syno, 1),
];

// deterministic seeded RNG wrapper: floats in [0,1], ints in [lo,hi]
public struct Rng
{
    private Mt19937 gen;

    public this(uint seed)
    {
        gen = Mt19937(seed);
    }

    public double unit()
    {
        return uniform(0.0, 1.0, gen);
    }

    public double below(double bound)
    {
        return uniform(0.0, bound, gen);
    }

    public int intIn(int lo, int hi)
    {
        return uniform(lo, hi + 1, gen);
    }
}

// one tagged text span inside a component
public struct Seg
{
    string text;
    string label;
}

// comma-separated component of the address string
public struct Comp
{
    Seg[] segs;
}

// one tagged example
public struct Example
{
    string[] tokens;
    string[] tags;
}

// the address units of one example: filled from an addresses
// row (real components) or sampled from the buckets
// (synthetic rows); compose renders them per the grammar
public struct Components
{
    Seg[] door;
    string road;
    Locality[] localities;
    string city;
    string postcode;

    // premises detail between the door and the road
    string house;
    string unit;
    string level;
}

// components → tagged example: pick a skeleton from templates.yaml
// (address_generator.templates), render it, then noise + comma
// dropout
public Example compose(Components c, Rng* rng, Buckets b,
    bool noise)
{
    TmplCtx ctx;
    ctx.rng = rng;
    ctx.b = b;
    ctx.c = c;

    auto comps = renderTemplate(pickSkeleton(c, rng), ctx);

    // train-only noise
    if (noise && rng.unit() < P_LOWERCASE)
    {
        foreach (ref comp; comps)
            foreach (ref s; comp.segs)
                s.text = s.text.toLower();
    }

    dropCommas(comps, rng);

    return tagExample(comps);
}

// state threaded through one renderTemplate pass: slot
// occurrence counters + last road text
private struct RenderState
{
    int[string] seen;
    string roadText;
}

// the immutable inputs of one compose: the row, the buckets,
// the rng
private struct TmplCtx
{
    Rng* rng;
    Buckets b;
    Components c;
}

private enum Sep { none, comma, glue, glueForce }

// render a template into comps: slots fill via fillSlot, ","
// breaks the comp, "~"/"+" glue into it ("~" falls back to a
// comma under MIN_ROW_TOKENS), literals render O. Skipped or
// empty slots vanish together with their separator
private Comp[] renderTemplate(string tmpl, ref TmplCtx ctx)
{
    Comp[] comps;
    Seg[] cur;
    auto pending = Sep.none;

    // render-local: slot occurrence counters + last road text
    RenderState rs;

    size_t i;

    while (i < tmpl.length)
    {
        const char ch = tmpl[i];

        if (ch == '{')
        {
            const auto close = i + countUntil(tmpl[i .. $], '}');
            const auto name = tmpl[i + 1 .. close];
            i = close + 1;

            auto segs = fillSlot(name, rs, ctx);

            if (segs.length == 0)
                continue;

            const bool split = pending == Sep.comma
                || (pending == Sep.glue
                    && !glueFits(ctx, comps, cur, segs));

            if (split && cur.length > 0)
            {
                comps ~= Comp(cur);
                cur = [];
            }

            cur ~= segs;
            pending = Sep.none;
        }
        else if (ch == ',')
        {
            pending = Sep.comma;
            i++;
        }
        else if (ch == '~')
        {
            pending = Sep.glue;
            i++;
        }
        else if (ch == '+')
        {
            pending = Sep.glueForce;
            i++;
        }
        else
            i = pushLiteral(tmpl, i, cur, pending);
    }

    if (cur.length > 0)
        comps ~= Comp(cur);

    return comps;
}

// a literal run up to the next slot/separator renders O inside
// the current comp; whitespace-only runs drop. Returns the
// index past the run
private size_t pushLiteral(string tmpl, size_t i, ref Seg[] cur,
    ref Sep pending)
{
    const auto start = i;

    while (i < tmpl.length && tmpl[i] != '{' && tmpl[i] != ','
            && tmpl[i] != '~' && tmpl[i] != '+')
        i++;

    const auto lit = tmpl[start .. i];
    bool ws = true;

    foreach (c; lit)
        if (!isWhite(c))
        {
            ws = false;
            break;
        }

    if (!ws)
    {
        cur ~= Seg(lit, "O");
        pending = Sep.none;
    }

    return i;
}

// the "~" floor: gluing drops the comma between the comps, the
// row must stay at MIN_ROW_TOKENS content tokens (the same
// floor as dropCommas)
private bool glueFits(ref TmplCtx ctx, Comp[] comps, Seg[] cur,
    Seg[] segs)
{
    size_t tokens = contentTokens(comps) + tailTokens(ctx.c);

    foreach (s; cur)
        tokens += words(s.text).length;

    foreach (s; segs)
        tokens += words(s.text).length;

    return tokens >= MIN_ROW_TOKENS;
}

// definite content tokens after the glue point (city,
// postcode, later localities) — the "~" MIN_ROW_TOKENS floor
// counts them before they render
private size_t tailTokens(Components c)
{
    size_t tokens = words(c.city).length + words(c.postcode).length;

    foreach (i, loc; c.localities)
        if (i > 0)
            tokens += words(loc.name).length;

    return tokens;
}

// one template slot → segs; occurrence n picks the n-th value
// of a repeatable slot (road, locality)
private Seg[] fillSlot(string name, ref RenderState rs,
    ref TmplCtx ctx)
{
    const auto n = rs.seen.get(name, 0);
    rs.seen[name] = n + 1;

    auto c = ctx.c;

    switch (name)
    {
    case "door":
        return n == 0 ? c.door : [];

    case "unit":
    case "level":
    case "house":
        return fillPremises(name, n, ctx);

    case "complex":
        // no locality vocab, no residential complex
        return n > 0 || ctx.b.localities.length == 0
            ? [] : fillComplex(ctx);

    case "road":
        return fillRoad(n, rs, ctx);

    case "locality":
        if (n >= c.localities.length)
            return [];

        return [Seg(recase(ctx.rng, c.localities[n].name), "suburb")];

    case "landmark":
        // the skeleton decides: only landmark templates carry the
        // slot, a second occurrence never renders; no POI vocab,
        // no landmark
        if (n > 0 || ctx.b.pois.length == 0)
            return [];

        return landmarkComp(ctx.rng, ctx.b).segs;

    case "city":
        return n == 0 && c.city.length > 0 ? [Seg(c.city, "city")] : [];

    case "postcode":
        return n == 0 && c.postcode.length > 0
            ? [Seg(c.postcode, "postcode")] : [];

    default:
        return [];
    }
}

// unit/level/house: the row value when present, a bucket or
// synthesized pick otherwise
private Seg[] fillPremises(string name, size_t n, ref TmplCtx ctx)
{
    if (n > 0)
        return [];

    auto c = ctx.c;

    switch (name)
    {
    case "unit":
        return c.unit.length > 0 ? [Seg(c.unit, "unit")]
            : pickUnit(ctx.rng, ctx.b);

    case "level":
        if (c.level.length > 0)
            return [Seg(levelPhrase(c.level), "level")];

        auto lv = pickLevel(ctx.rng, ctx.b);

        return lv.length > 0 ? [Seg(lv, "level")] : [];

    default: // "house"
        if (c.house.length > 0)
            return [Seg(c.house, "house")];

        auto h = pickHouse(ctx.rng, ctx.b);

        return h.length > 0 ? [Seg(h, "house")] : [];
    }
}

// the pair is all-or-nothing: an area colliding with a row
// locality drops both spans; one comp, the comma rides inside
// the area span
private Seg[] fillComplex(ref TmplCtx ctx)
{
    auto area = pickArea(ctx.rng, ctx.b, ctx.c);

    if (area.length == 0)
        return [];

    return [Seg(pickBuilding(ctx.rng, ctx.b), "house"),
        Seg(", " ~ area, "suburb")];
}

private Seg[] fillRoad(size_t n, ref RenderState rs, ref TmplCtx ctx)
{
    auto c = ctx.c;

    if (n == 0)
    {
        if (c.road.length == 0)
            return [];

        auto text = restyleRoad(ctx.rng, c.road);
        rs.roadText = text;

        return [Seg(text, "road")];
    }

    // a roadless row stays roadless — roadText is only set
    // when a road actually rendered, nothing to pair
    if (rs.roadText.length == 0)
        return [];

    auto second = pickSecondRoad(rs, ctx);
    rs.roadText = second;

    return [Seg(second, "road")];
}

// second road value: a real bucket road half the time, a
// synthesized Main/Cross otherwise; never a repeat of the
// first road. Ordinal second roads carry the case mix too
private string pickSecondRoad(ref RenderState rs, ref TmplCtx ctx)
{
    auto rng = ctx.rng;
    string second;

    if (ctx.b.roads.length > 0 && rng.unit() < P_REAL_SECOND_ROAD)
    {
        second = ctx.b.roads[rng.intIn(
            0, cast(int) ctx.b.roads.length - 1)].name;

        if (second == rs.roadText)
            second = synthRoad(rng);
    }
    else
        second = synthRoad(rng);

    if (isOrdinalRoad(second))
        second = recase(rng, second);

    return second;
}

// trade comma dropout: merge adjacent comps so tagExample
// renders no comma between them ("J P Nagar Bangalore");
// labels stay separate spans
private void dropCommas(ref Comp[] comps, Rng* rng)
{
    // content tokens of the whole row, commas excluded:
    // the fully-commaless rendering holds exactly these
    if (contentTokens(comps) < MIN_ROW_TOKENS)
        return;

    const bool rowStyle = rng.unit() < P_ROW_COMMALESS;

    Comp[] merged;

    foreach (c; comps)
    {
        const bool drop = merged.length > 0
            && (rowStyle || rng.unit() < P_BOUNDARY_COMMALESS);

        if (drop)
            merged[$ - 1].segs ~= c.segs;
        else
            merged ~= c;
    }

    comps = merged;
}

// content tokens of the row, commas excluded — the
// fully-commaless rendering holds exactly these
private size_t contentTokens(Comp[] comps)
{
    size_t tokens;

    foreach (c; comps)
        foreach (s; c.segs)
            tokens += words(s.text).length;

    return tokens;
}

// "<marker> <poi>" component; marker words stay O
private Comp landmarkComp(Rng* rng, Buckets b)
{
    auto marker = pickMarker(rng);
    string poi;

    if (rng.unit() < P_REAL_NAME_LANDMARK)
        poi = b.pois[rng.intIn(0, cast(int) b.pois.length - 1)];
    else
        poi = GENERIC_LANDMARKS[rng.intIn(0, GENERIC_LANDMARKS.length - 1)];

    return Comp([Seg(marker, "O"), Seg(recase(rng, poi), "landmark")]);
}

// caps/title/lower mix for landmark and locality spans
public string recase(Rng* rng, string s)
{
    auto p = rng.unit();

    if (p < P_CASE_CAPS)
        return s.toUpper();

    if (p < P_CASE_CAPS + P_CASE_LOWER)
        return s.toLower();

    return titleCase(s);
}

// trade renders spaced initials dotted or glued; OSM spells
// them spaced ("J C Nagar" -> "J.C.Nagar" / "JC Nagar")
private enum P_INITIAL_DOTTED = 0.30;
private enum P_INITIAL_GLUED = 0.10;

public string dottedInitials(string s)
{
    return joinInitials(s, ".", ".");
}

public string gluedInitials(string s)
{
    return joinInitials(s, "", " ");
}

// any single-letter word followed by another word, spaced or
// dotted ("St." is a word, not an initial)
public bool hasInitialRun(string s)
{
    auto parts = split(normalizeDots(s));

    foreach (i, w; parts)
        if (isInitial(w) && i + 1 < parts.length)
            return true;

    return false;
}

// compose-time mix for real-row localities: the native
// spelling stays most of the time. Synth rows pick their
// spelling deterministically instead (localityVariants)
public string restyleInitials(Rng* rng, string s)
{
    if (!hasInitialRun(s))
        return s;

    auto p = rng.unit();

    if (p < P_INITIAL_DOTTED)
        return dottedInitials(s);

    if (p < P_INITIAL_DOTTED + P_INITIAL_GLUED)
        return gluedInitials(s);

    return s;
}

// ordinal streets ("11th Cross", "2ND MAIN") carry the case
// mix: trade writes them caps, OSM titles them — the road
// label must not hinge on casing (impute hand50: caps
// ordinals leaked to unit/house once the caps bulk-import
// street labels left with imputation)
public string restyleRoad(Rng* rng, string s)
{
    s = restyleInitials(rng, s);

    return isOrdinalRoad(s) ? recase(rng, s) : s;
}

// "11th Cross", "10TH A CROSS": the first word is digits
// plus an ordinal suffix. ASCII-only — byte-slicing a
// multibyte codepoint would decode mid-sequence
private bool isOrdinalRoad(string s)
{
    auto parts = split(s);

    if (parts.length == 0 || parts[0].length < 3)
        return false;

    auto f = parts[0];

    foreach (c; f)
        if (c >= 0x80)
            return false;

    return isDigits(f[0 .. $ - 2])
        && canFind(["st", "nd", "rd", "th"],
            f[$ - 2 .. $].toLower());
}

private bool isInitial(string w)
{
    return w.length == 1 && isAlpha(w[0]);
}

// glue the words after a single-letter one: sepInitial joins
// the initial run, sepHead the first word after it
private string joinInitials(string s, string sepInitial,
    string sepHead)
{
    auto parts = split(normalizeDots(s));
    string r;

    foreach (i, w; parts)
    {
        if (i == 0)
        {
            r = w;
            continue;
        }

        auto prev = parts[i - 1];

        if (isInitial(prev))
            r ~= (isInitial(w) ? sepInitial : sepHead) ~ w;
        else
            r ~= " " ~ w;
    }

    return r;
}

// turn initial dots into spaces so the name splits into words
// uniformly ("A.C.Nagar" -> "A C Nagar"); other dots survive
// ("St. Mary" keeps its dot)
public string normalizeDots(string s)
{
    string r;

    foreach (i, char c; s)
    {
        if (c == '.' && i > 0 && isAlpha(s[i - 1])
                && (i < 2 || !isAlpha(s[i - 2])))
        {
            r ~= ' ';
            continue;
        }

        r ~= c;
    }

    return r;
}

// "KAMMANAHALLI MAIN ROAD" → "Kammanahalli Main Road": first
// letter of every space-separated word up, the rest down
private string titleCase(string s)
{
    string r;
    bool head = true;

    foreach (dchar c; s)
    {
        if (c == ' ')
        {
            r ~= ' ';
            head = true;
            continue;
        }

        r ~= head ? toUpper(c) : toLower(c);
        head = false;
    }

    return r;
}

// residential area name: a vocab locality the row does not
// already carry; empty when every draw collides
private string pickArea(Rng* rng, Buckets b, Components c)
{
    foreach (i; 0 .. AREA_DRAWS)
    {
        auto loc = b.localities[rng.intIn(
            0, cast(int) b.localities.length - 1)];

        if (!isRowLocality(c, loc.name))
            return loc.name;
    }

    return "";
}

private bool isRowLocality(Components c, string name)
{
    foreach (loc; c.localities)
        if (loc.name == name)
            return true;

    return false;
}

// compound building name: a real addr:housename half the
// time, a synthesized tower style otherwise. Never POI
// names: the POI bucket is parks and hospitals, POI names
// carry the landmark label, and libpostal's generator maps
// building names to house
private string pickBuilding(Rng* rng, Buckets b)
{
    if (b.housenames.length > 0 && rng.unit() < P_REAL_BUILDING)
        return b.housenames[rng.intIn(
            0, cast(int) b.housenames.length - 1)];

    return synthBuilding(rng);
}

// "Tower C" style: kind + one letter
public string synthBuilding(Rng* rng)
{
    auto kind = BUILDING_KINDS[rng.intIn(0, BUILDING_KINDS.length - 1)];

    return format!"%s %s"(kind, randUpper(rng));
}

// unit segs from the bucket, or a synthesized shop/flat style:
// row and bucket values pass whole, synthesized ones split the
// kind marker off (O) like doorSegs splits decorations
private Seg[] pickUnit(Rng* rng, Buckets b)
{
    if (rng.unit() >= P_UNIT)
        return [];

    if (b.units.length > 0 && rng.unit() < P_REAL_UNIT)
        return [Seg(b.units[rng.intIn(
            0, cast(int) b.units.length - 1)], "unit")];

    return synthUnit(rng);
}

private string pickLevel(Rng* rng, Buckets b)
{
    if (rng.unit() >= P_LEVEL)
        return "";

    // real levels first, the uniform floor list as fallback
    if (b.levels.length > 0)
        return levelPhrase(b.levels[rng.intIn(
            0, cast(int) b.levels.length - 1)]);

    return LEVELS[rng.intIn(0, LEVELS.length - 1)];
}

// OSM level=* is a bare number ("2", "0", "-1"): render it
// in the floor dialect the trade data carries; word values
// ("ground", "mezzanine") pass through
public string levelPhrase(string v)
{
    v = stripSpaces(v);

    if (v.length == 0)
        return "";

    if (v == "0")
        return "Ground Floor";

    if (v == "-1")
        return "Basement";

    if (isBareNumber(v))
        return ordinal(to!int(v)) ~ " Floor";

    return v;
}

// "2", "11": digits only, at most three
private bool isBareNumber(string v)
{
    if (v.length == 0 || v.length > 3)
        return false;

    foreach (c; v)
        if (!isDigit(c))
            return false;

    return true;
}

// housename: bucket-only, there is no honest way to
// synthesize one
private string pickHouse(Rng* rng, Buckets b)
{
    if (b.housenames.length == 0 || rng.unit() >= P_HOUSE)
        return "";

    return b.housenames[rng.intIn(0, cast(int) b.housenames.length - 1)];
}

// "<kind> <code>" style: the kind word marks the unit the way
// NEAR marks a landmark, only the code carries the label —
// trade evals grade the code ("Unit No. 23 B" → unit "23 B")
private Seg[] synthUnit(Rng* rng)
{
    auto code = unitCode(rng);

    // bare trade codes carry a letter ("S22"): bare numbers
    // read as doors ("84, M.G. Road"), so only lettered codes
    // open a row unitless
    if (hasLetter(code) && rng.unit() < P_UNIT_BARE)
        return [Seg(code, "unit")];

    auto kind = UNIT_KINDS[rng.intIn(0, UNIT_KINDS.length - 1)];

    // hyphen marker: "Shop No-23-B" fuses kind to the code
    if (kind.canFind("No.") && rng.unit() < P_UNIT_KIND_HYPHEN)
        kind = kind[0 .. $ - 1] ~ "-";

    return [Seg(kind, "O"), Seg(code, "unit")];
}

private bool hasLetter(string s)
{
    foreach (c; s)
        if (isAsciiLetter(c))
            return true;

    return false;
}

// trade unit codes: bare numbers, letter-suffixed ("23 B"),
// letter-prefixed ("R4"), letter-fused-suffix ("R1A") and
// hyphenated ("23-B") shapes from hand-eval data
private string unitCode(Rng* rng)
{
    // same-prefix pair: "R1 & R1A" (trade pairs share the letter)
    if (rng.unit() < P_UNIT_PAIR)
    {
        auto n1 = rng.intIn(1, UNIT_SUB_MAX).to!string;
        auto n2 = rng.intIn(1, UNIT_SUB_MAX).to!string;
        auto l = randUpper(rng);
        auto tail = rng.unit() < 0.5 ? randUpper(rng) : "";

        return l ~ n1 ~ " & " ~ l ~ n2 ~ tail;
    }

    auto n = rng.intIn(1, UNIT_SUB_MAX).to!string;

    switch (rng.intIn(0, 4))
    {
        case 0: return n;
        case 1: return n ~ " " ~ randUpper(rng);
        case 2: return randUpper(rng) ~ n;
        case 3: return n ~ "-" ~ randUpper(rng);
        default: return randUpper(rng) ~ n ~ randUpper(rng);
    }
}

private string pickMarker(Rng* rng)
{
    int total;

    foreach (m; MARKERS)
        total += m.weight;

    auto r = rng.below(total);

    foreach (m; MARKERS)
    {
        r -= m.weight;

        if (r <= 0)
            return m.word;
    }

    return MARKERS[$ - 1].word;
}

public string synthRoad(Rng* rng)
{
    // "5th Main" style street, sometimes with a Road suffix
    auto kind = rng.intIn(0, 1) == 0 ? "Main" : "Cross";
    auto road = format!"%s %s"(ordinal(rng.intIn(1, ORD_MAX)), kind);

    if (rng.unit() < P_ROAD_SUFFIX)
        road ~= " Road";

    return road;
}

private string ordinal(int n)
{
    if (n % 100 >= 11 && n % 100 <= 13)
        return to!string(n) ~ "th";

    auto suffix = ["th", "st", "nd", "rd"][n % 10 <= 3 ? n % 10 : 0];

    return to!string(n) ~ suffix;
}

// door decorations precede the number: "No.", "#", "G"/"W-",
// ward letter-code "TJ7P-", "SY NO", "SurveyNo.". Downstream
// geocoders (photon, nominatim) match on the clean number, so
// the decoration stays untagged and only the number carries
// house_number. Ranges ("7-3-272") and plot doors ("12/7")
// start with a digit and pass through whole.
public Seg[] doorSegs(string val)
{
    string prefix, core;

    if (!splitDoor(val, prefix, core))
        return [Seg(val, "house_number")];

    return [Seg(prefix, "O"), Seg(core, "house_number")];
}

private bool isAsciiLetter(char c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

// "No. 12" -> "No."/"12", "#12" -> "#"/"12", "TJ7P-12" ->
// "TJ7P-"/"12"; false when the value holds no decoration
private bool splitDoor(string s, out string prefix, out string core)
{
    prefix = "";
    core = "";

    if (s.length == 0 || s[0].isDigit)
        return false;

    // "#12"
    if (s[0] == '#')
    {
        auto i = 1;

        while (i < s.length && s[i] == ' ')
            i++;

        if (i < s.length && s[i].isDigit)
        {
            prefix = "#";
            core = s[i .. $];
            return true;
        }

        return false;
    }

    size_t i;

    while (i < s.length && isAsciiLetter(s[i]))
        i++;

    // decoration needs at least one letter: "-12" is not one
    if (i == 0)
        return false;

    // ward letter-code: "TJ7P-" = letters, digit, letters, '-'
    if (i < s.length && s[i].isDigit)
    {
        auto j = i + 1;

        while (j < s.length && isAsciiLetter(s[j]))
            j++;

        if (j < s.length && s[j] == '-' && j + 1 < s.length
                && s[j + 1].isDigit)
        {
            prefix = s[0 .. j + 1];
            core = s[j + 1 .. $];
            return true;
        }
    }

    // word decoration: "No.", "SY NO", "SurveyNo.", "Plot No:"
    auto end = i;

    while (end < s.length && !s[end].isDigit)
    {
        if (isAsciiLetter(s[end]) || canFind(". - :;+", s[end]))
        {
            end++;
            continue;
        }

        return false;
    }

    // no number after the decoration ("Kerala")
    if (end == s.length)
        return false;

    prefix = stripSpaces(s[0 .. end]);
    core = s[end .. $];

    return true;
}

// " No." → "No.": strip spaces at both ends
private string stripSpaces(string s)
{
    size_t a, b = s.length;

    while (a < b && s[a] == ' ')
        a++;

    while (b > a && s[b - 1] == ' ')
        b--;

    return s[a .. b];
}

private DoorStyle pickDoorStyle(Rng* rng)
{
    int total;

    foreach (w; DOOR_WEIGHTS)
        total += w.weight;

    auto r = rng.below(total);

    foreach (w; DOOR_WEIGHTS)
    {
        r -= w.weight;

        if (r <= 0)
            return w.style;
    }

    return DOOR_WEIGHTS[$ - 1].style;
}

// pick a style and render: full door restyle in one call
public string restyleDoor(string val, Rng* rng)
{
    return renderDoor(pickDoorStyle(rng), val, rng);
}

// "12" → "12", "12/7", "No. 12", "#12", "G12"/"W-12",
// "TJ7P-12" (ward letter-code prefix)
private string renderDoor(DoorStyle s, string val, Rng* rng)
{
    final switch (s)
    {
    case DoorStyle.plain:
        return val;

    case DoorStyle.plot:
        return val ~ "/" ~ rng.intIn(1, PLOT_SUB_MAX).to!string;

    case DoorStyle.no:
        return "No. " ~ val;

    case DoorStyle.hash:
        return "#" ~ val;

    // survey-number trade dialect; doorSegs splits it like
    // every other decoration: "SY NO" stays O
    case DoorStyle.syno:
        return "SY NO " ~ val;

    case DoorStyle.alpha:
        return randUpper(rng)
            ~ (rng.unit() < 0.5 ? "-" : "") ~ val;

    case DoorStyle.ward:
        return randUpper(rng) ~ randUpper(rng)
            ~ rng.intIn(0, 9).to!string ~ randUpper(rng)
            ~ "-" ~ val;
    }
}

// one uppercase letter from ALPHA
private string randUpper(Rng* rng)
{
    auto i = cast(size_t) rng.intIn(0, ALPHA.length - 1);

    return ALPHA[i .. i + 1];
}

// Python \w = letters + numbers + '_' (str.isalnum semantics).
// isAlpha also counts combining marks (Other_Alphabetic vowel
// signs, anusvaras), but Python does not: marks split runs and
// become single-char tokens
// ("തേക്കടി" → ത േ ക ് കട ി)
private bool isWordChar(dchar c)
{
    if (c == '_')
        return true;

    if (isMark(c))
        return false;

    return isAlpha(c) || isNumber(c);
}

// \w+ | [^\w\s]: runs of unicode alphanumerics/underscore, or
// one punctuation character per token
public string[] words(string text)
{
    string[] toks;
    dchar[] word;

    void flush()
    {
        if (word.length > 0)
        {
            toks ~= to!string(word);
            word.length = 0;
        }
    }

    foreach (dchar c; text)
    {
        if (isWordChar(c))
        {
            word ~= c;
            continue;
        }

        flush();

        if (c.isWhite)
            continue;

        toks ~= to!string([c]);
    }

    flush();

    return toks;
}

// components → tokens + BIO tags, comma separators stay O
public Example tagExample(Comp[] comps)
{
    string[] tokens, tags;

    foreach (i, c; comps)
    {
        if (i > 0)
        {
            tokens ~= ",";
            tags ~= "O";
        }

        foreach (s; c.segs)
        {
            bool started;

            foreach (w; words(s.text))
            {
                // embedded comma inside a field value: separator,
                // not part of the span; next word starts a new B- span
                if (w == ",")
                {
                    tokens ~= ",";
                    tags ~= "O";
                    started = false;
                    continue;
                }

                string tag = "O";

                if (s.label != "O")
                {
                    if (started)
                        tag = "I-" ~ s.label;
                    else
                    {
                        tag = "B-" ~ s.label;
                        started = true;
                    }
                }

                tokens ~= w;
                tags ~= tag;
            }
        }
    }

    return Example(tokens, tags);
}

unittest
{
    // word runs, punctuation per token, whitespace dropped
    assert(words("12/3, MG Road") == ["12", "/", "3", ",", "MG", "Road"]);
    assert(words("No. 7") == ["No", ".", "7"]);
    assert(words("4th Cross") == ["4th", "Cross"]);

    // Python \w excludes combining marks: vowel signs, viramas
    // and anusvaras split runs and stand alone as tokens
    assert(words("എറണാകുളം")
        == ["എറണ", "ാ", "ക", "ു", "ള", "ം"]);
    assert(words("തേക്കടി") == ["ത", "േ", "ക", "്", "കട", "ി"]);
    assert(words("೫೬೦೦೬೦") == ["೫೬೦೦೬೦"]);

    // 1st/2nd/3rd, 11th-13th, everything else th
    assert(ordinal(1) == "1st");
    assert(ordinal(2) == "2nd");
    assert(ordinal(3) == "3rd");
    assert(ordinal(11) == "11th");
    assert(ordinal(112) == "112th");
    assert(ordinal(21) == "21st");
    assert(ordinal(30) == "30th");
}

unittest
{
    // door formats: fixed style renders deterministically, the
    // original number is preserved
    auto rng = Rng(42);

    assert(renderDoor(DoorStyle.plain, "12", &rng) == "12");
    assert(renderDoor(DoorStyle.no, "12", &rng) == "No. 12");
    assert(renderDoor(DoorStyle.hash, "12", &rng) == "#12");
    assert(renderDoor(DoorStyle.syno, "12", &rng) == "SY NO 12");

    auto plot = renderDoor(DoorStyle.plot, "12", &rng);
    assert(plot[0 .. 3] == "12/" && isDigits(plot[3 .. $]));

    auto alpha = renderDoor(DoorStyle.alpha, "12", &rng);
    assert(ALPHA.canFind(alpha[0]) && alpha.canFind("12"));

    auto ward = renderDoor(DoorStyle.ward, "12", &rng);
    assert(ALPHA.canFind(ward[0]) && ALPHA.canFind(ward[1]));
    assert(isDigits(ward[2 .. 3]) && ALPHA.canFind(ward[3]));
    assert(ward[4 .. $] == "-12");
}

private bool isDigits(string v)
{
    foreach (dchar c; v)
        if (!isNumber(c))
            return false;

    return v.length > 0;
}

unittest
{
    // clean-number doors: no decoration, one span
    assert(doorSegs("12") == [Seg("12", "house_number")]);
    assert(doorSegs("12/7") == [Seg("12/7", "house_number")]);
    assert(doorSegs("35/1A") == [Seg("35/1A", "house_number")]);

    // Hyderabad ranges are numbers, not decorations
    assert(doorSegs("7-3-272") == [Seg("7-3-272", "house_number")]);

    // decorated doors: decoration O, clean number tagged
    assert(doorSegs("No. 12")
        == [Seg("No.", "O"), Seg("12", "house_number")]);
    assert(doorSegs("No 12/3")
        == [Seg("No", "O"), Seg("12/3", "house_number")]);
    assert(doorSegs("#12")
        == [Seg("#", "O"), Seg("12", "house_number")]);
    assert(doorSegs("G12")
        == [Seg("G", "O"), Seg("12", "house_number")]);
    assert(doorSegs("W-12")
        == [Seg("W-", "O"), Seg("12", "house_number")]);
    assert(doorSegs("TJ7P-12")
        == [Seg("TJ7P-", "O"), Seg("12", "house_number")]);
    assert(doorSegs("SY NO 40")
        == [Seg("SY NO", "O"), Seg("40", "house_number")]);
    assert(doorSegs("SurveyNo.132P")
        == [Seg("SurveyNo.", "O"), Seg("132P", "house_number")]);
    assert(doorSegs("Plot No. 12")
        == [Seg("Plot No.", "O"), Seg("12", "house_number")]);
    assert(doorSegs("Ward No: 24")
        == [Seg("Ward No:", "O"), Seg("24", "house_number")]);
    assert(doorSegs("ward ; 30")
        == [Seg("ward ;", "O"), Seg("30", "house_number")]);
    assert(doorSegs("B + 6")
        == [Seg("B +", "O"), Seg("6", "house_number")]);

    // lowercase doors (noise augmentation) and letter-only or
    // non-ASCII values stay whole
    assert(doorSegs("no. 12")
        == [Seg("no.", "O"), Seg("12", "house_number")]);
    assert(doorSegs("Kerala") == [Seg("Kerala", "house_number")]);

    // tokens/tags: only the number is B-house_number, the
    // decoration stays O
    auto ex = tagExample([Comp(doorSegs("No. 12")),
        Comp([Seg("MG Road", "road")])]);

    assert(ex.tokens == ["No", ".", "12", ",", "MG", "Road"]);
    assert(ex.tags
        == ["O", "O", "B-house_number", "O", "B-road", "I-road"]);

    // glued tokens: locality follows the road directly, no
    // comma, labels stay separate spans
    auto glued = tagExample([Comp([Seg("80 Feet Road", "road"),
        Seg("Rajendra Nagara", "suburb")])]);

    assert(glued.tokens
        == ["80", "Feet", "Road", "Rajendra", "Nagara"]);
    assert(glued.tags == ["B-road", "I-road", "I-road",
        "B-suburb", "I-suburb"]);
}

unittest
{
    // glue floor: a single-word street plus a single-word
    // locality is 2 content tokens — the glued form ("Cross
    // Kammanahalli") falls under MIN_TOKENS and the dataset
    // loop drops the row (coverage pairs lose ~700 rows per
    // generation that way). Tiny rows keep the comma ("Cross,
    // Kammanahalli"), the same floor as dropCommas
    auto b = Buckets([], [], [], [], [], [], [], []);

    foreach (seed; 0 .. 256)
    {
        auto rng = Rng(seed);

        Components c;
        c.road = "Cross";
        c.localities ~= Locality("Kammanahalli", "suburb");

        auto ex = compose(c, &rng, b, true);

        assert(ex.tokens.length >= MIN_ROW_TOKENS);

        // labels stay separate spans either way
        assert(ex.tags.canFind("B-road"));
        assert(ex.tags.canFind("B-suburb"));
    }
}

unittest
{
    // front placement: the trade street field also carries a
    // comma-separated locality in front of a road — the
    // single-road frame ("HULIMAVU, B G ROAD") and the
    // multi-road frame ("Gear School Road, Panathur, Outer
    // Ring Road"). Front = before the LAST road comp, so the
    // multi-road row keeps road1 in front; with a city the
    // default branch would sit the locality behind road2
    auto b = Buckets([Road("Outer Ring Road")], [], [], [],
        [], [], [], []);

    bool single, multi;

    foreach (seed; 0 .. 1024)
    {
        auto rng = Rng(seed);

        auto c = Components(doorSegs("2"), "Gear School Road",
            [Locality("Panathur", "suburb")], "Bangalore",
            "560036", "", "", "5");

        auto ex = compose(c, &rng, b, false);

        ptrdiff_t iSub = -1, iRoad = -1, iRoad2 = -1,
            iCity = -1;

        foreach (i, t; ex.tags)
        {
            if (t == "B-suburb" && iSub == -1)
                iSub = i;

            if (t == "B-road" && iRoad == -1)
                iRoad = i;

            if (t == "B-road" && iRoad != -1 && i > iRoad
                && iRoad2 == -1)
                iRoad2 = i;

            if (t == "B-city" && iCity == -1)
                iCity = i;
        }

        if (iSub != -1 && iSub < iRoad)
            single = true;

        if (iRoad2 != -1 && iRoad < iSub && iSub < iRoad2
            && iRoad2 < iCity)
            multi = true;
    }

    assert(single);
    assert(multi);
}

unittest
{
    // premises detail: unit, level and housename comps sit
    // between the door and the road ("12, Shop No. 1,
    // 2nd Floor, Shanti Nivas, MG Road"), in that order
    auto b = Buckets([], [], [], [], [], [], [], []);

    auto rng = Rng(42);

    auto c = Components(doorSegs("12"), "MG Road", [],
        "Bangalore", "560034", "Shanti Nivas", "Shop No. 1",
        "2nd Floor");

    auto ex = compose(c, &rng, b, false);

    ptrdiff_t iDoor = -1, iUnit = -1, iLevel = -1, iHouse = -1,
        iRoad = -1;

    foreach (i, t; ex.tags)
    {
        if (t == "B-house_number" && iDoor == -1)
            iDoor = i;

        if (t == "B-unit" && iUnit == -1)
            iUnit = i;

        if (t == "B-level" && iLevel == -1)
            iLevel = i;

        if (t == "B-house" && iHouse == -1)
            iHouse = i;

        if (t == "B-road" && iRoad == -1)
            iRoad = i;
    }

    assert(iDoor < iUnit && iUnit < iLevel && iLevel < iHouse
        && iHouse < iRoad);

    // injection: buckets supply units and housenames at trade
    // rates, floors are synthesized
    auto bb = Buckets([Road("MG Road")], [],
        [Locality("Koramangala", "suburb")], ["Bangalore"],
        ["560034"], ["Shanti Nivas"], ["Shop No. 1"], []);

    int unit, level, house;

    foreach (seed; 0 .. 128)
    {
        auto srng = Rng(seed);

        auto sc = Components(doorSegs("12"), "MG Road", [],
            "Bangalore", "560034");

        auto sex = compose(sc, &srng, bb, false);

        foreach (t; sex.tags)
        {
            unit += t == "B-unit";
            level += t == "B-level";
            house += t == "B-house";
        }
    }

    assert(unit > 0 && level > 0 && house > 0);
}

unittest
{
    // synthesized units: the kind word marks the unit like NEAR
    // marks a landmark, only the code carries the label
    // ("Unit No. 23 B" → O marker + unit "23 B") — trade evals
    // grade the code; doorSegs splits decorations the same way
    int withUnitNo;

    foreach (seed; 0 .. 64)
    {
        auto rng = Rng(seed);
        auto segs = synthUnit(&rng);

        // bare trade code ("S22"): the unit opens the row,
        // only the code carries the label, and the code always
        // carries a letter ("84" alone is a door)
        if (segs.length == 1)
        {
            assert(segs[0].label == "unit");
            assert(hasLetter(segs[0].text));

            assert(tagExample([Comp(segs)]).tags[0]
                == "B-unit");

            continue;
        }

        assert(segs.length == 2);
        assert(segs[0].label == "O");

        // kind words, plain ("Unit No.") or hyphen-fused
        // ("Shop No-")
        assert(UNIT_KINDS.canFind(segs[0].text)
            || segs[0].text.canFind("No-"));
        assert(segs[1].label == "unit");

        // tokens: kind words O, code B-unit
        auto ex = tagExample([Comp(segs)]);

        assert(ex.tags[0] == "O");
        assert(ex.tags.canFind("B-unit"));

        if (segs[0].text == "Unit No.")
            withUnitNo++;
    }

    assert(withUnitNo > 0);
}

unittest
{
    // trade unit codes beyond the three base shapes: hyphenated
    // ("23-B"), letter-fused suffix ("R1A") and same-prefix
    // pairs ("R1 & R1A"); every class must fire across seeds
    int plain, suffixed, prefixed, hyphen, fusedSuf, pair;

    foreach (seed; 0 .. 512)
    {
        auto rng = Rng(seed);
        auto code = unitCode(&rng);

        if (code.canFind(" & "))
        {
            pair = true;
            continue;
        }

        if (isDigits(code))
            plain = true;
        else if (code.canFind("-"))
        {
            auto parts = code.split("-");
            hyphen += parts.length == 2 && isDigits(parts[0])
                && isAsciiLetter(parts[1][0])
                && parts[1].length == 1;
        }
        else if (isAsciiLetter(code[0])
            && isAsciiLetter(code[$ - 1])
            && isDigits(code[1 .. $ - 1]))
            fusedSuf = true;
        else if (isAsciiLetter(code[0])
            && isDigits(code[1 .. $]))
            prefixed = true;
        else if (code[$ - 2] == ' ' && code[0 .. $ - 2].isDigits)
            suffixed = true;
    }

    assert(plain > 0);
    assert(suffixed > 0);
    assert(prefixed > 0);
    assert(hyphen > 0);
    assert(fusedSuf > 0);
    assert(pair > 0);
}

unittest
{
    // bare trade code ("S22, Mantri Mall ..."): the unit opens
    // the row with no kind marker — only the code carries the
    // label; the marked kind style must still fire, and the
    // hyphen marker variant ("Shop No-23-B") must appear
    int bare, marked, hyphenKind;

    foreach (seed; 0 .. 512)
    {
        auto rng = Rng(seed);
        auto segs = synthUnit(&rng);

        if (segs.length == 1)
        {
            bare++;

            assert(segs[0].label == "unit");
            assert(hasLetter(segs[0].text));

            auto ex = tagExample([Comp(segs)]);

            assert(ex.tags[0] == "B-unit");
            assert(!ex.tags.canFind("O"));

            continue;
        }

        marked++;

        assert(segs.length == 2);
        assert(segs[0].label == "O");
        assert(segs[1].label == "unit");

        if (segs[0].text.canFind("No-"))
            hyphenKind++;
    }

    assert(bare > 0);
    assert(marked > 0);
    assert(hyphenKind > 0);
}

unittest
{
    // compose-level: an injected synthetic unit keeps its kind
    // words untagged — the O marker sits directly before the
    // B-unit code, same comp, no comma between
    auto b = Buckets([], [], [], [], [], [], [], []);
    int fired;

    foreach (seed; 0 .. 2000)
    {
        auto rng = Rng(seed);

        auto c = Components(doorSegs("12"), "MG Road", [],
            "Bangalore", "560034", "", "", "");

        auto ex = compose(c, &rng, b, false);

        ptrdiff_t iUnit = -1;

        foreach (i, t; ex.tags)
            if (t == "B-unit")
            {
                iUnit = i;
                break;
            }

        if (iUnit == -1)
            continue;

        fired++;

        // marked kind: the O kind word sits directly before the
        // code, no comma; bare code: its comp opener (comma) or
        // a merged door tag under comma dropout; landmark O
        // markers ("NEAR") also read O
        assert(ex.tokens[iUnit - 1] == ","
            || ex.tags[iUnit - 1].canFind("house_number")
            || ex.tags[iUnit - 1] == "O");
    }

    assert(fired > 0);
}

unittest
{
    // OSM level=* is a bare number: compose renders it in
    // the floor dialect the trade data carries ("2" -> "2nd
    // Floor"); word values pass through
    assert(levelPhrase("2") == "2nd Floor");
    assert(levelPhrase("0") == "Ground Floor");
    assert(levelPhrase("11") == "11th Floor");
    assert(levelPhrase("-1") == "Basement");
    assert(levelPhrase("Ground") == "Ground");
    assert(levelPhrase("") == "");

    // a row level renders inside the level comp, between the
    // door and the road
    auto b = Buckets([], [], [], [], [], [], [], []);

    auto rng = Rng(42);

    auto c = Components(doorSegs("12"), "MG Road", [],
        "Bangalore", "560034", "", "", "2");

    auto ex = compose(c, &rng, b, false);

    ptrdiff_t il = -1;

    foreach (i, t; ex.tags)
        if (t == "B-level")
        {
            il = i;
            break;
        }

    assert(il != -1);

    assert(ex.tokens[il] == "2nd");
    assert(ex.tokens[il + 1] == "Floor");
    assert(ex.tags[il + 1] == "I-level");

    // levels bucket feeds injection
    auto bb = Buckets([], [], [], [], [], [], [], ["3"]);

    int level;

    foreach (seed; 0 .. 128)
    {
        auto srng = Rng(seed);

        auto sc = Components(doorSegs("12"), "MG Road", [],
            "Bangalore", "560034", "", "", "");

        auto sex = compose(sc, &srng, bb, false);

        foreach (t; sex.tags)
            level += t == "B-level";
    }

    assert(level > 0);
}

unittest
{
    // residential compound: "<building>, <area>" between the
    // premises detail and the road; the area never
    // duplicates a locality the row already carries. Empty
    // housenames: the building is always a synth style, so
    // the row carries exactly one house span
    auto b = Buckets([Road("Brookefield Road")], [],
        [Locality("Divyasree Republic of Whitefield", "suburb"),
            Locality("Kundalahalli", "suburb")],
        ["Bengaluru"], ["560066"], [], [], []);

    int compound;

    foreach (seed; 0 .. 64)
    {
        auto rng = Rng(seed);

        auto c = Components(doorSegs("G4"), "Brookefield Road",
            [Locality("Kundalahalli", "suburb")],
            "Bengaluru", "560066", "", "", "");

        auto ex = compose(c, &rng, b, false);

        // the pair is all-or-nothing: one building + one area
        int nBuilding;

        foreach (t; ex.tags)
            nBuilding += t == "B-house";

        if (nBuilding == 0)
            continue;

        assert(nBuilding == 1);

        compound++;

        // span order: door, building, area, road, locality;
        // the area is the first suburb span after the building
        ptrdiff_t iDoor = -1, iBuilding = -1, iArea = -1,
            iRoad = -1;

        foreach (i, t; ex.tags)
        {
            if (t == "B-house_number" && iDoor == -1)
                iDoor = i;

            if (t == "B-house" && iBuilding == -1)
                iBuilding = i;

            if (t == "B-suburb" && iArea == -1
                && iBuilding != -1 && i > iBuilding)
                iArea = i;

            if (t == "B-road" && iRoad == -1)
                iRoad = i;
        }

        assert(iDoor < iBuilding && iBuilding < iArea);

        if (iRoad != -1)
            assert(iArea < iRoad);

        // the area skips the row's own locality: with one
        // other vocab entry it is always that one
        assert(ex.tokens[iArea] == "Divyasree");
    }

    assert(compound > 0);

    // synthesized building names: "<kind> <letter>"
    auto rng = Rng(7);

    auto s = synthBuilding(&rng);

    assert(canFind(BUILDING_KINDS, s[0 .. $ - 2]));
    assert(ALPHA.canFind(s[$ - 1]));
}

unittest
{
    // the compound building is a house span drawn from
    // addr:housename or synth styles, never a POI name — the
    // POI bucket is parks and hospitals ("Sobha Jasmine Park"
    // is leisure=park), POI names carry the landmark label,
    // and libpostal's own generator maps building names to
    // house (osm/formatter.py: name, addr:housename → HOUSE)
    auto b = Buckets([Road("Brookefield Road")], ["Wadur-II"],
        [Locality("Divyasree Republic of Whitefield", "suburb"),
            Locality("Kundalahalli", "suburb")],
        ["Bengaluru"], ["560066"], ["Shanti Nivas"], [], []);

    int compound;

    foreach (seed; 0 .. 128)
    {
        auto rng = Rng(seed);

        auto c = Components(doorSegs("G4"), "Brookefield Road",
            [Locality("Kundalahalli", "suburb")],
            "Bengaluru", "560066", "", "", "");

        auto ex = compose(c, &rng, b, false);

        foreach (i, t; ex.tags)
        {
            // the building label is gone
            assert(t != "B-building" && t != "I-building", t);

            // POI names only ever carry the landmark label
            if (ex.tokens[i] == "Wadur")
                assert(t == "B-landmark", t);

            if (t != "B-house")
                continue;

            // join the whole house span
            string span;

            foreach (j; i .. ex.tags.length)
            {
                if (j > i && ex.tags[j] != "I-house")
                    break;

                if (j > i)
                    span ~= " ";

                span ~= ex.tokens[j];
            }

            const bool fromVocab = span == "Shanti Nivas";

            const bool synth = span.length >= 3
                && canFind(BUILDING_KINDS, span[0 .. $ - 2])
                && ALPHA.canFind(span[$ - 1]);

            assert(fromVocab || synth, span);

            compound++;
        }
    }

    assert(compound > 0);
}

unittest
{
    // compose: the trade shape "door, <road> <locality>,
    // <city>, <pin>" and its comma-separated variant both
    // occur; every example carries door, road and exactly
    // one locality span
    auto b = Buckets([Road("80 Feet Road")], ["Canara Bank"],
        [Locality("Rajendra Nagara", "suburb")],
        ["Bangalore"], ["560034"], [], [], []);

    int glued, separate, multiRoad, landmark;

    foreach (seed; 0 .. 32)
    {
        auto rng = Rng(seed);

        auto c = Components(doorSegs("12"), "80 Feet Road",
            [Locality("Rajendra Nagara", "suburb")],
            "Bangalore", "560034", "", "", "");

        auto ex = compose(c, &rng, b, false);

        assert(ex.tokens.length == ex.tags.length);

        // exactly one locality span
        int spans;

        foreach (t; ex.tags)
            spans += t == "B-suburb";

        assert(spans == 1);

        bool door, road;

        foreach (i, t; ex.tags)
        {
            door = door || t == "B-house_number";
            road = road || t == "B-road";

            if (t == "B-suburb" && i > 0)
            {
                if (ex.tokens[i - 1] != "," && ex.tags[i - 1].canFind("road"))
                    glued++;
                else if (ex.tokens[i - 1] == ",")
                    separate++;
            }
        }

        // multi-road and landmark variants occur across seeds
        int roadSpans;

        foreach (t; ex.tags)
            roadSpans += t == "B-road";

        multiRoad += roadSpans >= 2;

        foreach (t; ex.tags)
            if (t == "B-landmark")
            {
                landmark++;
                break;
            }

        assert(door && road);
    }

    assert(glued > 0 && separate > 0);
    assert(multiRoad > 0);
    assert(landmark > 0);

    // no street on the row: roads come as a synthesized
    // Main/Cross pair or not at all; the door stays first
    foreach (seed; 0 .. 16)
    {
        auto rng = Rng(seed);

        auto c = Components(doorSegs("12"), "",
            [Locality("Rajendra Nagara", "suburb")],
            "Bangalore", "560034", "", "", "");

        auto ex = compose(c, &rng, b, false);

        int roadSpans;

        foreach (t; ex.tags)
            roadSpans += t == "B-road";

        assert(roadSpans == 0 || roadSpans >= 2);
        assert(ex.tags[0] == "B-house_number");
    }
}

unittest
{
    // trade comma dropout: fields join without the comma at
    // random ("J P Nagar Bangalore"); whole rows run
    // comma-free (writer style). Labels stay separate spans
    auto b = Buckets([], [], [], [], [], [], [], []);

    int commaCity, commalessCity, commaFreeRows;

    foreach (seed; 0 .. 64)
    {
        auto rng = Rng(seed);

        auto c = Components(doorSegs("12"), "80 Feet Road", [],
            "Bangalore", "560034", "", "", "");

        auto ex = compose(c, &rng, b, false);

        assert(ex.tokens.length == ex.tags.length);

        auto ci = ex.tags.countUntil("B-city");

        // the city span is always present, preceded by a
        // comma token or directly by the road span — never
        // anything else
        assert(ci > 0);

        if (ex.tokens[ci - 1] == ",")
        {
            commaCity++;
            assert(ex.tags[ci - 1] == "O");
        }
        else
        {
            commalessCity++;
            assert(ex.tags[ci - 1].canFind("road"));
        }

        commaFreeRows += !ex.tokens.canFind(",");
    }

    assert(commaCity > 0);
    assert(commalessCity > 0);
    assert(commaFreeRows > 0);

    // tiny rows keep every comma: dataset.d drops examples
    // below MIN_TOKENS = 3, and the commaless rendering
    // must not push rows under that floor ("12, Bangalore"
    // survives, "12 Bangalore" would not)
    foreach (seed; 0 .. 32)
    {
        auto rng = Rng(seed);

        auto comps = [Comp([Seg("12", "house_number")]),
            Comp([Seg("Bangalore", "city")])];

        dropCommas(comps, &rng);

        auto ex = tagExample(comps);

        assert(ex.tokens == ["12", ",", "Bangalore"]);
    }
}

unittest
{
    // case augmentation: landmark and locality spans render in
    // caps/title/lower mixes — the label must attach to the
    // marker and the position, not to a casing heuristic
    // ("KAMMANAHALLI" was tagged road for being caps,
    // "NEAR MINERVA MILL" for not being in the vocab)
    assert(titleCase("KAMMANAHALLI") == "Kammanahalli");
    assert(titleCase("R V ENGINEERING COLLEGE")
        == "R V Engineering College");
    assert(titleCase("Minerva Mill") == "Minerva Mill");

    int caps, title, lower;

    foreach (seed; 0 .. 256)
    {
        auto rng = Rng(seed);

        auto s = recase(&rng, "Minerva Mill");

        if (s == "MINERVA MILL")
            caps++;
        else if (s == "Minerva Mill")
            title++;
        else if (s == "minerva mill")
            lower++;
        else
            assert(0, s);
    }

    assert(caps > 0 && title > 0 && lower > 0);

    // composed locality spans carry the mix too, glued and
    // separate alike; the recased text stays one span
    auto b = Buckets([Road("80 Feet Road")], [],
        [Locality("Kammanahalli", "suburb")], ["Bangalore"],
        ["560034"], [], [], []);

    int locCaps, locLower;

    foreach (seed; 0 .. 512)
    {
        auto rng = Rng(seed);

        auto c = Components(doorSegs("12"), "80 Feet Road",
            [Locality("Kammanahalli", "suburb")], "Bangalore",
            "560034", "", "", "");

        auto ex = compose(c, &rng, b, false);

        ptrdiff_t il = -1;

        foreach (i, t; ex.tags)
            if (t == "B-suburb")
            {
                il = i;
                break;
            }

        assert(il != -1);

        if (ex.tokens[il] == "KAMMANAHALLI")
            locCaps++;
        else if (ex.tokens[il] == "kammanahalli")
            locLower++;
        else
            assert(ex.tokens[il] == "Kammanahalli");

        // the recased name is one span, whatever the case
        ptrdiff_t end = il;

        while (end + 1 < ex.tags.length && ex.tags[end + 1] == "I-suburb")
            end++;

        for (ptrdiff_t k = il; k <= end; k++)
            assert(ex.tokens[k] != ",");
    }

    assert(locCaps > 0 && locLower > 0);
}

unittest
{
    // landmark position augmentation: after the roads (default),
    // glued to the road comp (trade street field, no comma),
    // right after the door (trade door field), in the locality
    // slot before the city (trade area field), or at the very
    // end past the postcode
    auto b = Buckets([Road("80 Feet Road")], ["Water Tank"],
        [Locality("Kammanahalli", "suburb")], ["Bangalore"],
        ["560034"], [], [], []);

    int afterRoad, glued, afterDoor, areaSlot, end;

    foreach (seed; 0 .. 4096)
    {
        auto rng = Rng(seed);

        auto c = Components(doorSegs("12"), "80 Feet Road",
            [Locality("Kammanahalli", "suburb")], "Bangalore",
            "560034", "", "", "");

        auto ex = compose(c, &rng, b, false);

        ptrdiff_t iLand = -1, iRoad = -1, iLoc = -1,
            iPost = -1;

        foreach (i, t; ex.tags)
        {
            if (t == "B-landmark" && iLand == -1)
                iLand = i;

            if (t == "B-road" && iRoad == -1)
                iRoad = i;

            if (t == "B-suburb" && iLoc == -1)
                iLoc = i;

            if (t == "B-postcode" && iPost == -1)
                iPost = i;
        }

        if (iLand == -1)
            continue;

        // glued: the marker O tokens sit between the road and
        // the POI name — walk back over them to the road
        auto j = iLand;

        while (j > 0 && ex.tags[j - 1] == "O"
                && ex.tokens[j - 1] != ",")
            j--;

        if (j > 0 && ex.tokens[j - 1] != ","
                && ex.tags[j - 1].canFind("road"))
            glued++;
        else if (iLand > iPost)
            end++;
        else if (iLand < iRoad)
            afterDoor++;
        else if (iLand > iLoc)
            areaSlot++;
        else
            afterRoad++;
    }

    assert(afterRoad > 0 && glued > 0 && afterDoor > 0);
    assert(areaSlot > 0 && end > 0);
}

unittest
{
    // trade renders spaced initials dotted or glued; OSM
    // spells them spaced. The transforms normalize native
    // dotted names ("A.C Nagar") too
    assert(dottedInitials("J C Nagar") == "J.C.Nagar");
    assert(dottedInitials("A K Nagar Colony") == "A.K.Nagar Colony");
    assert(dottedInitials("A.C Nagar") == "A.C.Nagar");
    assert(dottedInitials("Anchu Thayil Colony S N Puram P O")
        == "Anchu Thayil Colony S.N.Puram P.O");
    assert(gluedInitials("J C Nagar") == "JC Nagar");
    assert(gluedInitials("K G Nagara") == "KG Nagara");
    assert(gluedInitials("A.C Nagar") == "AC Nagar");

    // names without initial runs pass through; "St." is a
    // word, not an initial
    assert(dottedInitials("Kammanahalli") == "Kammanahalli");
    assert(dottedInitials("St. Mary Road") == "St. Mary Road");
    assert(gluedInitials("St. Mary Road") == "St. Mary Road");

    assert(hasInitialRun("J C Nagar"));
    assert(hasInitialRun("A.C Nagar"));
    assert(!hasInitialRun("Kammanahalli"));
    assert(!hasInitialRun("St. Mary Road"));
}

unittest
{
    // restyleInitials mix: native spelling stays most of the
    // time, dotted and glued both appear
    int dotted, glued, spaced;

    foreach (seed; 0 .. 512)
    {
        auto rng = Rng(seed);

        auto s = restyleInitials(&rng, "J C Nagar");

        if (s == "J.C.Nagar")
            dotted++;
        else if (s == "JC Nagar")
            glued++;
        else
        {
            assert(s == "J C Nagar", s);
            spaced++;
        }
    }

    assert(dotted > 0 && glued > 0 && spaced > 0);

    // names without initial runs come back untouched
    auto rng = Rng(7);

    assert(restyleInitials(&rng, "Kammanahalli") == "Kammanahalli");
    assert(restyleInitials(&rng, "St. Mary Road") == "St. Mary Road");
}

unittest
{
    // compose renders a locality verbatim: every spelling stays
    // one span, dots never split it — the spelling choice is
    // dataset policy, not grammar
    auto b = Buckets([Road("15th Cross")], [],
        [Locality("J C Nagar", "suburb")], ["Bangalore"],
        ["560086"], [], [], []);

    foreach (spell; ["J C Nagar", "J.C.Nagar", "JC Nagar"])
    {
        int seen;

        foreach (seed; 0 .. 64)
        {
            auto rng = Rng(seed);

            auto c = Components(doorSegs("16"), "15th Cross",
                [Locality(spell, "suburb")], "Bangalore",
                "560086", "", "", "");

            auto ex = compose(c, &rng, b, false);

            // the compound area also renders B-suburb and sits
            // before the road; the row locality is the last one
            ptrdiff_t il = -1;

            foreach (i, t; ex.tags)
                if (t == "B-suburb")
                    il = i;

            assert(il != -1);

            ptrdiff_t end = il;

            while (end + 1 < ex.tags.length
                    && ex.tags[end + 1] == "I-suburb")
                end++;

            // the span is contiguous, commas never inside it
            foreach (k; il .. end + 1)
                assert(ex.tokens[k] != ",");

            auto span = join(ex.tokens[il .. end + 1], " ").toLower();

            assert(span == "j c nagar"
                || span == "j . c . nagar" || span == "jc nagar",
                span);
            assert(span == join(words(spell), " ").toLower());

            seen++;
        }

        assert(seen == 64);
    }
}

unittest
{
    // roadless rows stay roadless: with no street in the
    // components the row renders no road span — the multi-road
    // augmentation must not fabricate a Main/Cross pair.
    // Trade and hand-eval rows carry roadless addresses
    // ("26/1, ORION MALL, INDUSTRIAL SUBURB, I BLOCK,
    // RAJAJINAGAR, 560010"); the tagger must see absence
    auto b = Buckets([Road("MG Road")], ["Canara Bank"],
        [Locality("Koramangala", "suburb")], ["Bangalore"],
        ["560034"], [], [], []);

    foreach (seed; 0 .. 128)
    {
        auto rng = Rng(seed);

        auto c = Components(doorSegs("12"), "", [], "Bangalore",
            "560034", "", "", "");

        auto ex = compose(c, &rng, b, true);

        foreach (t; ex.tags)
            assert(t != "B-road",
                format("seed %s: %s", seed, ex.tokens));

        // the row still renders: door, city and pin survive
        bool door, city, pin;

        foreach (t; ex.tags)
        {
            door = door || t == "B-house_number";
            city = city || t == "B-city";
            pin = pin || t == "B-postcode";
        }

        assert(door && city && pin,
            format("seed %s: %s", seed, ex.tokens));
    }
}

unittest
{
    // the noise pass lowercases but never reorders: slot
    // order is the grammar's contract, a permuted row teaches
    // an impossible address (a road-first "L G Complex" row
    // traced to P_SHUFFLE). Empty buckets keep the row bare:
    // door, road, city, postcode
    auto b = Buckets([], [], [], ["Bangalore"], ["560034"],
        [], [], []);

    foreach (seed; 0 .. 128)
    {
        auto rng = Rng(seed);

        auto c = Components(doorSegs("12/7"), "80 Feet Road",
            [], "Bangalore", "560034", "", "", "");

        auto ex = compose(c, &rng, b, true);

        ptrdiff_t iDoor = -1, iRoad = -1, iCity = -1,
            iPost = -1;

        foreach (i, t; ex.tags)
        {
            if (t == "B-house_number" && iDoor == -1)
                iDoor = i;

            if (t == "B-road" && iRoad == -1)
                iRoad = i;

            if (t == "B-city" && iCity == -1)
                iCity = i;

            if (t == "B-postcode" && iPost == -1)
                iPost = i;
        }

        assert(iDoor < iRoad && iRoad < iCity && iCity < iPost,
            format("seed %s: %s", seed, ex.tokens));
    }
}

unittest
{
    // trade hyphen city-PIN ("BENGALURU - 560038"): one comp,
    // the hyphen stays O, the spans stay separate — the
    // boundary must not fuse the PIN into the city span
    // (impute hand50: city "BENGALURU 560038", postcode lost).
    // Both renderings occur across seeds
    auto b = Buckets([], [], [], [], [], [], [], []);

    int hyphen, separate;

    foreach (seed; 0 .. 256)
    {
        auto rng = Rng(seed);

        auto c = Components(doorSegs("12"), "80 Feet Road", [],
            "Bengaluru", "560038", "", "", "");

        auto ex = compose(c, &rng, b, false);

        auto ci = ex.tags.countUntil("B-city");
        auto pi = ex.tags.countUntil("B-postcode");

        assert(ci > 0 && pi > ci);

        if (ex.tokens[pi - 1] == "-")
        {
            hyphen++;

            assert(ex.tags[pi - 1] == "O");
            assert(ex.tokens[ci + 1] == "-");
            assert(ex.tags[ci + 1] == "O");
            continue;
        }

        separate++;
    }

    assert(hyphen > 0 && separate > 0);
}

unittest
{
    // template engine: a repeated slot consumes successive
    // values — "{road}, {road}" renders a two-road street, the
    // second never repeats the first
    auto b = Buckets([Road("MG Road")], [], [], [], [], [], [], []);

    TmplCtx ctx;
    auto rng = Rng(42);
    ctx.rng = &rng;
    ctx.b = b;
    ctx.c.road = "80 Feet Road";

    auto ex = tagExample(renderTemplate("{road}, {road}", ctx));

    assert(ex.tokens.length == ex.tags.length);

    int roads;

    foreach (t; ex.tags)
        roads += t == "B-road";

    assert(roads == 2);
    assert(ex.tokens[0] == "80" && ex.tokens[4] != "80");
}

unittest
{
    // "~" glues the next slot into the previous comp (no comma,
    // labels stay separate spans); under MIN_ROW_TOKENS it
    // falls back to a comma — the same floor as dropCommas
    auto b = Buckets([], [], [], [], [], [], [], []);

    TmplCtx ctx;
    auto rng = Rng(3);
    ctx.rng = &rng;
    ctx.b = b;
    ctx.c.road = "80 Feet Road";
    ctx.c.localities ~= Locality("Rajendra Nagara", "suburb");

    auto glued = tagExample(renderTemplate("{road}~{locality}", ctx));

    assert(!glued.tokens.canFind(","));
    assert(glued.tags == ["B-road", "I-road", "I-road",
        "B-suburb", "I-suburb"]);

    // tiny row: "Cross" + "Kammanahalli" is 2 content tokens,
    // the glue must fall back to the comma
    TmplCtx tiny;
    auto rng2 = Rng(3);
    tiny.rng = &rng2;
    tiny.b = b;
    tiny.c.road = "Cross";
    tiny.c.localities ~= Locality("Kammanahalli", "suburb");

    auto split = tagExample(renderTemplate("{road}~{locality}", tiny));

    assert(split.tokens == ["Cross", ",", "Kammanahalli"]);
    assert(split.tags == ["B-road", "O", "B-suburb"]);

    // "+" glues unconditionally, even under the floor
    TmplCtx force;
    auto rng3 = Rng(3);
    force.rng = &rng3;
    force.b = b;
    force.c.road = "Cross";
    force.c.localities ~= Locality("Kammanahalli", "suburb");

    auto forced = tagExample(renderTemplate("{road}+{locality}", force));

    assert(forced.tokens == ["Cross", "Kammanahalli"]);
}

unittest
{
    // literals render O inside the comp: "{city} - {postcode}"
    // is one comp, the hyphen a lone O token between the spans
    auto b = Buckets([], [], [], [], [], [], [], []);

    TmplCtx ctx;
    auto rng = Rng(11);
    ctx.rng = &rng;
    ctx.b = b;
    ctx.c.city = "Bengaluru";
    ctx.c.postcode = "560038";

    auto ex = tagExample(renderTemplate("{city} - {postcode}", ctx));

    assert(ex.tokens == ["Bengaluru", "-", "560038"]);
    assert(ex.tags == ["B-city", "O", "B-postcode"]);

    // empty slots vanish together with their separator
    TmplCtx e;
    auto rng2 = Rng(11);
    e.rng = &rng2;
    e.b = b;
    e.c.road = "MG Road";
    e.c.city = "Bangalore";

    auto ex2 = tagExample(renderTemplate("{door}, {unit}, {road}, {city}", e));

    assert(ex2.tokens == ["MG", "Road", ",", "Bangalore"]);
    assert(ex2.tags == ["B-road", "I-road", "O", "B-city"]);
}

unittest
{
    // ordinal streets carry the case mix: trade writes caps
    // ("11TH CROSS"), OSM titles them — the road label must
    // not hinge on casing (impute hand50: caps ordinals
    // leaked to unit/house). Second roads recase too ("3rd
    // Main, 5TH CROSS"). Non-ordinal roads keep their
    // native spelling
    auto b = Buckets([Road("5th Cross")], [], [], [], [], [], [], []);

    int caps, title, lower, secondCaps;

    foreach (seed; 0 .. 2048)
    {
        auto rng = Rng(seed);

        auto c = Components(doorSegs("12"), "11th Cross", [],
            "Bengaluru", "560038", "", "", "");

        auto ex = compose(c, &rng, b, false);

        // the row road is the first B-road in the stream
        auto ri = ex.tags.countUntil("B-road");

        if (ex.tokens[ri] == "11TH")
        {
            assert(ex.tokens[ri + 1] == "CROSS");
            assert(ex.tags[ri + 1] == "I-road");
            caps++;
        }
        else
        {
            assert(ex.tokens[ri] == "11th");

            auto k = ex.tokens[ri + 1];

            assert(k == "Cross" || k == "cross");
            assert(ex.tags[ri + 1] == "I-road");
            title += k == "Cross";
            lower += k == "cross";
        }

        // second roads recase too: the bucket road "5th Cross"
        // renders caps somewhere across seeds
        foreach (i, t; ex.tags)
            if (t == "B-road" && i != ri && ex.tokens[i] == "5TH")
                secondCaps++;
    }

    assert(caps > 0 && title > 0 && lower > 0);
    assert(secondCaps > 0);

    // non-ordinal roads pass through untouched
    foreach (seed; 0 .. 64)
    {
        auto rng = Rng(seed);

        assert(restyleRoad(&rng, "MG Road") == "MG Road");
        assert(restyleRoad(&rng, "80 Feet Road")
            == "80 Feet Road");
    }

    // non-ASCII first words never slice into a codepoint
    assert(!isOrdinalRoad("ಕ್ರಾಸ್ ರೋಡ್"));
    assert(!isOrdinalRoad("क रोड Cross"));
}
