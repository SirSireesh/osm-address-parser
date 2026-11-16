// row skeletons as data: templates.yaml (OpenCage-style {{slot}}
// placeholders, probability + requires annotations), embedded at
// compile time via import() and parsed once with dyaml. compose()
// picks a skeleton per row from this list; the renderer's slot
// syntax is {slot}, so load rewrites {{slot}} → {slot}
module address_generator.templates;

import std.algorithm : map, sum;
import std.algorithm.searching : canFind;
import std.array : array;
import std.conv : to;
import std.string : replace;

import dyaml;

import address_generator.grammar : Components, Rng;

public struct RowTemplate
{
    string skeleton;
    double weight;
    string[] requires;
}

private enum string TEMPLATES_YAML = import("address_generator/templates.yaml");

private RowTemplate[] g_templates;

// parsed once, on first use
public RowTemplate[] templates()
{
    if (g_templates.length > 0)
        return g_templates;

    auto root = Loader.fromString(TEMPLATES_YAML).load();

    foreach (dyaml.Node t; root["templates"])
    {
        RowTemplate rt;
        rt.skeleton = t["template"].as!string
            .replace("{{", "{").replace("}}", "}");
        rt.weight = t["probability"].as!double;

        if (auto req = "requires" in t)
            foreach (dyaml.Node r; *req)
                rt.requires ~= r.as!string;

        g_templates ~= rt;
    }

    return g_templates;
}

// the row's present components, by requires-name
private bool has(string req, const Components c)
{
    switch (req)
    {
    case "road": return c.road.length > 0;
    case "locality": return c.localities.length > 0;
    case "city": return c.city.length > 0;
    case "postcode": return c.postcode.length > 0;
    default: return true;
    }
}

// weighted pick among the templates whose requires the row
// satisfies; falls back to the unconditional base (index 0).
// The {{localities}} placeholders expand to one {locality} slot
// per row locality: {{localities}} = all of them, {{localities_more}}
// = all but the first (for skeletons that place the first locality
// separately, glued or in front)
public string pickSkeleton(const Components c, Rng* rng)
{
    auto all = templates();

    RowTemplate[] eligible;
    double total = 0;

    foreach (t; all)
    {
        bool ok = true;

        foreach (r; t.requires)
            if (!has(r, c))
            {
                ok = false;
                break;
            }

        if (ok)
        {
            eligible ~= t;
            total += t.weight;
        }
    }

    if (eligible.length == 0)
        return expand(templates()[0].skeleton, c);

    auto pick = rng.unit() * total;

    foreach (t; eligible)
    {
        pick -= t.weight;

        if (pick <= 0)
            return expand(t.skeleton, c);
    }

    return expand(eligible[$ - 1].skeleton, c);
}

// {{localities}} → one {locality} slot per locality, comma-joined;
// {{localities_more}} → the same minus the first. Zero localities
// expand to nothing and the slot vanishes at render
private string expand(string skeleton, const Components c)
{
    string all, more;

    foreach (i; 0 .. c.localities.length)
    {
        all ~= (i > 0 ? ", " : "") ~ "{locality}";

        if (i > 0)
            more ~= ", {locality}";
    }

    return skeleton
        .replace("{localities}", all)
        .replace("{localities_more}", more);
}

unittest
{
    auto ts = templates();

    assert(ts.length > 0);
    assert(ts[0].skeleton == "{door}, {unit}, {level}, {house}, {road}, {localities}, {city}, {postcode}");

    // requires filter: a roadless row never gets the two-road shape
    auto rng = Rng(42);
    auto c = Components(null, "", [], "", "", "", "", "");

    foreach (i; 0 .. 200)
    {
        auto s = pickSkeleton(c, &rng);

        assert(!canFind(s, "{road}, {road}"));
    }
}

unittest
{
    import address_generator.buckets : Locality;

    int single, multi;

    foreach (seed; 0 .. 1024)
    {
        auto rng = Rng(cast(uint) seed);
        auto c = Components(null, "Gear School Road",
            [Locality("Panathur", "suburb")], "Bangalore", "560036",
            "", "", "");

        auto s = pickSkeleton(c, &rng);

        if (canFind(s, "{locality}, {road}"))
            single++;
        if (canFind(s, "{road}, {locality}, {road}"))
            multi++;
    }

    // front-placement and two-road-front shapes must be reachable
    assert(single > 0);
    assert(multi > 0);
}
