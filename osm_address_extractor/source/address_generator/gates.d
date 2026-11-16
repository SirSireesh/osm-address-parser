// Street-shape predicates shared by both generators:
// bucket_generator drops ordinal-only names at keep(),
// address_generator dedupes pairs and cross-filters street
// vocab with the same rules. One definition, identical
// semantics on both sides of the parquet boundary.
module address_generator.gates;

import std.algorithm.comparison : among;
import std.algorithm.searching : canFind;
import std.ascii : isAlpha, isDigit;
import std.string : endsWith, toLower;

// runs of alphanumeric characters, lowercased: "No.7" -> "no", "7"
public string[] tokens(string s)
{
	string[] words;
	size_t start = size_t.max;

	foreach (i, char c; s)
	{
		if (isAlpha(c) || isDigit(c))
		{
			if (start == size_t.max)
				start = i;

			continue;
		}

		if (start != size_t.max)
		{
			words ~= s[start .. i].toLower();
			start = size_t.max;
		}
	}

	if (start != size_t.max)
		words ~= s[start .. $].toLower();

	return words;
}

// "<n>th Cross", "<n>th Main Road": an ordinal plus one or two
// generic street words, and nothing else. Case-insensitive.
// Public: bucket_generator keep() drops these names,
// address_generator pair enumeration dedupes the same names
public bool isOrdinalStreet(string s)
{
	size_t i;

	// leading number, 1-3 digits
	size_t digits;
	while (i < s.length && isDigit(s[i]))
	{
		i++;
		digits++;
	}

	if (digits == 0 || digits > 3 || i + 2 > s.length)
		return false;

	if (!s[i .. i + 2].toLower().among("st", "nd", "rd", "th"))
		return false;

	i += 2;

	// "1st Cross", "2nd Main Road": cross/main plus an optional
	// cross/main/road/street suffix, and nothing else.
	// Block/stage/phase names stay: rarer in addresses, fine to
	// oversample. One or two words, single-space separated.
	foreach (n; 0 .. 2)
	{
		if (i == s.length)
			return n > 0;

		if (s[i] != ' ')
			return false;

		size_t start = ++i;
		while (i < s.length && s[i] != ' ')
			i++;

		auto w = s[start .. i].toLower();

		// cross/main plus an optional cross/main/road/street suffix
		if (n == 0 ? !w.among("cross", "main") : !w.among("cross", "main", "road", "street"))
			return false;
	}

	return i == s.length;
}

// locality words in a street name with no road word: the way
// carries its locality's name, not a street name ("L B S
// Colony", "sanath nagar") — a mislabel, not a road. Subword
// suffix match: bulk imports glue the word into one token
// ("PRATAPNAGAR", "Kuttipuram"); a road token anywhere saves
// the name ("Kammanahalli Main Road" keeps its substring
// "halli" behind the whole token "road")
private enum string[] LOCALITY_STREET_WORDS = [
	"nagar", "halli", "colony", "layout", "puram", "palya",
	"peta", "patti", "wada", "vattam", "agraharam", "medu",
	"hobli", "village", "town", "township", "block", "phase",
	"sector", "area",
];

private enum string[] STREET_ROAD_WORDS = [
	"road", "rd", "street",
];

public bool isLocalityStreet(string s)
{
	bool locality = false;

	foreach (w; tokens(s))
	{
		if (canFind(STREET_ROAD_WORDS, w))
			return false;

		foreach (loc; LOCALITY_STREET_WORDS)
			if (w.endsWith(loc))
				locality = true;
	}

	return locality;
}

unittest
{
	// ordinal-only cross/main names: skip
	assert(isOrdinalStreet("1st Cross"));
	assert(isOrdinalStreet("5th Main"));
	assert(isOrdinalStreet("2nd Cross Road"));
	assert(isOrdinalStreet("22ND MAIN"));
	assert(isOrdinalStreet("1st Main Road"));
	assert(isOrdinalStreet("10th Main Street"));
	assert(isOrdinalStreet("3rd Cross Street"));

	// rarer ordinals stay as vocab: block/stage/road/phase
	assert(!isOrdinalStreet("3rd Stage"));
	assert(!isOrdinalStreet("2nd Block"));
	assert(!isOrdinalStreet("4th Road"));
	assert(!isOrdinalStreet("1st Phase"));

	// anything beyond the ordinal pattern: keep
	assert(!isOrdinalStreet("1st Cross, Gandhi Bazaar"));
	assert(!isOrdinalStreet("Gandhi Bazaar 1st Cross"));
	assert(!isOrdinalStreet("Church Street"));
	assert(!isOrdinalStreet("Main Street"));
	assert(!isOrdinalStreet("1 Cross"));
	assert(!isOrdinalStreet("Cross"));
	assert(!isOrdinalStreet("12th Main Road East"));
	assert(!isOrdinalStreet("1stCross"));

	assert(isLocalityStreet("L B S Colony"));
	assert(isLocalityStreet("sanath nagar"));
	assert(isLocalityStreet("Anuvijay Township Layout"));
	assert(isLocalityStreet("Jakkur layout"));
	assert(isLocalityStreet("PRATAPNAGAR"));
	assert(isLocalityStreet("Kuttipuram"));
	assert(isLocalityStreet("V R Puram"));
	assert(isLocalityStreet("Industrial town"));
	assert(isLocalityStreet("K R Palya"));
	assert(isLocalityStreet("Venkatapeta"));

	// a road word saves it, whatever the locality word
	assert(!isLocalityStreet("S V Nagar Rd"));
	assert(!isLocalityStreet("Kammanahalli Main Road"));
	assert(!isLocalityStreet("Adoor - Vandiperiyar Highway"));
	assert(!isLocalityStreet("Aakriti Arv Viva"));
	assert(!isLocalityStreet("Chamarajpet"));
}
