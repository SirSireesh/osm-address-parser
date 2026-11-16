-- gates.sql — the bucket_generator gates as DuckDB macros.
--
-- ---------------- primitives ----------------

-- tags.get(key): '' instead of NULL for a missing key, the D
-- TagView semantics every gate below is written against.
-- NB: map[key] is the scalar lookup; map_extract() returns a
-- list of matches instead
CREATE OR REPLACE MACRO tag(tags, key) AS
    coalesce(tags[key], '');

-- runs of ASCII alphanumerics, lowercased (gates.d tokens):
-- "No.7" -> ["no", "7"]. Extract first, lower after — unicode
-- letters must not join runs by lowercasing into ASCII
CREATE OR REPLACE MACRO tokens(s) AS
    list_transform(regexp_extract_all(s, '[\p{N}\p{L}]+'), lambda t: lower(t));

-- ---------------- extract.d: isFullAddress ----------------

-- a maximal 6-digit run not starting with '0' (hasPin)
CREATE OR REPLACE MACRO has_pin(s) AS
    len(list_filter(regexp_extract_all(s, '[\p{N}]+'),
                    lambda r: len(r) = 6 AND r[1] != '0')) > 0;

-- '#' anywhere, digits-then-comma up front, or a no/nr/plot
-- token not preceded by 'road' (hasDoorNumber).
-- NB: textual macro expansion would re-run tokens(s) inside
-- the position lambda, ~2x per token per row — so the scan
-- runs once over the head token and once over adjacent
-- cur/prev pairs from a single tokens() zip
CREATE OR REPLACE MACRO has_door_number(s) AS
    contains(s, '#')
    OR regexp_matches(s, '^\p{N}+,')
    OR (len(tokens(s)) > 0 AND tokens(s)[1] IN ('no', 'nr', 'plot'))
    OR len(list_filter(
           list_zip(list_slice(tokens(s), 2, len(tokens(s))),
                    list_slice(tokens(s), 1, len(tokens(s)) - 1)),
           lambda p: p[1] IN ('no', 'nr', 'plot')
                  AND p[2] != 'road')) > 0;

-- the 'floor' word among the tokens (hasFloor)
CREATE OR REPLACE MACRO has_floor(s) AS
    list_contains(tokens(s), 'floor');

-- a digit/slash/digit plot id, as in "13-6-600/44" (hasPlotNumber)
CREATE OR REPLACE MACRO has_plot_number(s) AS
    regexp_matches(s, '\p{N}/\p{N}');

-- the whole addr:street is a postal address, not a street name
CREATE OR REPLACE MACRO is_full_address(s) AS
    has_pin(s) OR has_door_number(s) OR has_floor(s) OR has_plot_number(s);

-- ---------------- gates.d: isOrdinalStreet ----------------

-- "<n>th Cross", "<n>th Main Road": a 1-3 digit ordinal, then
-- one or two generic street words, and nothing else.
-- NB: the digit run stays ASCII (D std.ascii.isDigit)
CREATE OR REPLACE MACRO is_ordinal_street(s) AS
    regexp_matches(
        s,
        '^[0-9]{1,3}(st|nd|rd|th) (cross|main)( (cross|main|road|street))?$',
        'i');

-- ---------------- extract.d: keep ----------------

-- any addr:* key: TagView.has("addr:") is a prefix match
CREATE OR REPLACE MACRO has_addr(tags) AS
    len(list_filter(map_keys(tags), lambda k: starts_with(k, 'addr:'))) > 0;

-- which elements enter the pipeline: the address branch takes
-- every tagged address unless the street value is a full address
-- string; the rest keeps named place/highway/landuse/postal rows,
-- dropping ordinal-only street names
CREATE OR REPLACE MACRO keep(tags) AS
    CASE
        WHEN has_addr(tags) THEN NOT is_full_address(tag(tags, 'addr:street'))
        WHEN tag(tags, 'name') = '' THEN FALSE
        WHEN is_ordinal_street(tag(tags, 'name')) THEN FALSE
        ELSE tag(tags, 'place') != ''
            OR tag(tags, 'highway') != ''
            OR tag(tags, 'landuse') != ''
            OR tag(tags, 'boundary') = 'postal_code'
    END;

-- ---------------- extract.d: isAreaRelation ----------------

-- a relation that carries a polygon: a named place, or a named
-- admin area of level 8-10 or boundary type
CREATE OR REPLACE MACRO is_area_relation(tags) AS
    tag(tags, 'name') != ''
    AND (tag(tags, 'place') != ''
        OR (tag(tags, 'admin_level') != ''
            AND (tag(tags, 'admin_level') IN ('8', '9', '10')
                OR tag(tags, 'type') = 'boundary')));

-- ---------------- extract.d: cleanInvisible ----------------

-- format chars (ZWNJ/ZWSP/bidi) and exotic whitespace (NBSP,
-- tabs) never render: normalize to a space, never drop the
-- value. D's isWhite ∪ isFormat == \p{Z} ∪ \p{Cf} ∪ the C0
-- whitespace escapes and NEL; ' ' maps to itself. NB: the
-- 'g' flag — without it only the first match replaces
CREATE OR REPLACE MACRO clean_invisible(s) AS
    regexp_replace(s, '[\p{Cf}\p{Z}\t\n\v\f\r\x{85}]', ' ', 'g');

-- ---------------- extract.d: stripEdge ----------------

-- the edge-junk class: ' ,' plus noise punctuation. A '-'/'+'
-- before a digit is a sign, not junk: "-1", "+2" keep theirs
--
-- The D trim loop is a scan of the ORIGINAL string with a
-- forward-looking guard on both sides:
--   left:  stop before the first solid char, where solid =
--          outside the class, or a '-/+' with a digit after
--   right: stop after the last solid char, same predicate
-- RE2 has no lookahead, so each stop is the min/max of two
-- greedy scans: the class run, and the first/last sign-digit
-- pair ('[\s\S]' spans newlines, '.' would not)
CREATE OR REPLACE MACRO edge_left(s) AS
    least(
        len(regexp_extract(s, '^[ ,#*!?+_:;.@/"''-]*')),
        coalesce(
            len(nullif(regexp_extract(s, '^[\s\S]*?[-+]\p{N}'), '')) - 2,
            len(s)));

CREATE OR REPLACE MACRO edge_right(s) AS
    len(s) - 1 - greatest(
        len(s) - 1
            - len(regexp_extract(reverse(s), '^[ ,#*!?+_:;.@/"''-]*')),
        coalesce(
            len(nullif(regexp_extract(s, '[\s\S]*[-+]\p{N}'), '')) - 2,
            -1));

-- trim ' ,' and edge junk from both ends ("Foo ., ." → "Foo")
CREATE OR REPLACE MACRO strip_edge(s) AS
    CASE
        WHEN edge_left(s) + edge_right(s) >= len(s) THEN ''
        ELSE substring(s, edge_left(s) + 1,
                       len(s) - edge_left(s) - edge_right(s))
    END;

-- ---------------- extract.d: stripSep ----------------

-- " Foo, " → "Foo": strip spaces and commas at both ends
CREATE OR REPLACE MACRO strip_sep(s) AS
    trim(s, ' ,');

-- ---------------- extract.d: isJunkDoor ----------------

-- a junk door segment: empty, digitless (D reads Unicode
-- numbers), a floor phrase, or a bounded PIN inside
CREATE OR REPLACE MACRO is_junk_door_seg(s) AS
    s = ''
    OR NOT regexp_matches(s, '\p{N}')
    OR contains(lower(s), 'floor')
    OR has_pin(s);

-- bulk-import junk in the door column: a locality, a floor, a
-- PIN mash, or a comma cram. The wiki form ("12,14", no
-- spaces, every segment a door) survives; a letter run over 6
-- marks a glued street/place word
CREATE OR REPLACE MACRO is_junk_door(s) AS
    CASE
        WHEN NOT contains(s, ',') THEN is_junk_door_seg(strip_sep(s))
        ELSE contains(s, ' ')
            OR regexp_matches(s, '\p{L}{7,}')
            OR len(list_filter(string_split(s, ','),
                    lambda seg: is_junk_door_seg(strip_sep(seg)))) > 0
    END;

-- ---------------- extract.d: isFloorPhrase ----------------

-- "1st", "second": digit or word ordinal
CREATE OR REPLACE MACRO is_ordinal_word(w) AS
    w IN ('first', 'second', 'third', 'fourth', 'fifth', 'sixth',
          'seventh', 'eighth', 'ninth', 'tenth')
    OR (len(w) >= 3
        AND right(w, 2) IN ('st', 'nd', 'rd', 'th')
        AND regexp_full_match(left(w, len(w) - 2), '\p{N}+'));

-- a pure floor phrase: "Ground Floor", "3rd Floor", "Basement"
CREATE OR REPLACE MACRO is_floor_phrase(s) AS
    CASE len(tokens(s))
        WHEN 1 THEN tokens(s)[1] = 'basement'
        WHEN 2 THEN tokens(s)[2] = 'floor'
                AND (tokens(s)[1] = 'ground'
                     OR is_ordinal_word(tokens(s)[1]))
        ELSE FALSE
    END;

-- ---------------- extract.d: isJunkUnit ----------------

-- "1", "27A", "23B", "S22": 1-3 digits with at most 2 letters
-- glued on one edge; a bare letter is the "Wing C" code.
-- NB: letters stay ASCII (D std.ascii.isAlpha), digits are \p{N}
CREATE OR REPLACE MACRO is_code_token(t) AS
    regexp_full_match(t, '[A-Za-z]|[A-Za-z]{0,2}\p{N}{1,3}[A-Za-z]{0,2}');

-- token class of the unit gate, in the D check order.
-- NB: the init letters stay ASCII (D std.ascii.isAlpha)
CREATE OR REPLACE MACRO unit_cls(t) AS
    CASE
        WHEN is_ordinal_word(t) THEN 'ordinal'
        WHEN is_code_token(t) THEN 'code'
        WHEN t IN ('shop', 'no', 'unit', 'flat', 'suite', 'room', 'block',
                   'wing', 'tower', 'plot', 'office') THEN 'kind'
        WHEN len(t) <= 3 AND regexp_full_match(t, '[A-Za-z]+') THEN 'init'
        ELSE 'bad'
    END;

-- a unit only ever holds a unit code: 1-3 tokens, exactly one
-- code (two when the second is a bare-letter suffix), kind
-- words allowed, a short initialism only leading
CREATE OR REPLACE MACRO is_unit_code(s) AS
    len(tokens(s)) BETWEEN 1 AND 3
    AND NOT list_contains(list_transform(tokens(s), lambda t: unit_cls(t)),
                          'bad')
    AND NOT list_contains(list_transform(tokens(s), lambda t: unit_cls(t)),
                          'ordinal')
    AND (len(list_filter(tokens(s), lambda t: unit_cls(t) = 'code')) = 1
         OR (len(list_filter(tokens(s), lambda t: unit_cls(t) = 'code')) = 2
             AND list_contains(list_transform(tokens(s),
                     lambda t: unit_cls(t) = 'code' AND len(t) = 1), TRUE)))
    AND len(list_filter(range(len(tokens(s))), lambda i:
            unit_cls(tokens(s)[i + 1]) = 'init'
            AND len(list_filter(range(i), lambda j:
                    unit_cls(tokens(s)[j + 1]) = 'code')) > 0)) = 0;

-- roads, localities, buildings, doors, PIN mashes blank
CREATE OR REPLACE MACRO is_junk_unit(s) AS
    NOT is_unit_code(s);

-- ---------------- extract.d: cleanUnit ----------------

-- floors and junk leave the unit column: a pure floor phrase
-- fills an empty level slot, the rest blanks
CREATE OR REPLACE MACRO clean_unit(unit, level) AS
    CASE
        WHEN unit = '' THEN {'unit': '', 'level': ''}
        WHEN is_floor_phrase(unit)
            THEN {'unit': '',
                  'level': CASE WHEN level = '' THEN unit ELSE '' END}
        WHEN is_junk_unit(unit) THEN {'unit': '', 'level': ''}
        ELSE {'unit': unit, 'level': ''}
    END;

-- ---------------- extract.d: landmark directions ----------------

-- street column: the marker at any position junks the value
CREATE OR REPLACE MACRO is_direction_street(s) AS
    len(list_intersect(tokens(s), [
        'opp', 'opposite', 'near', 'behind', 'beside',
        'next', 'towards', 'across', 'below'])) > 0;

-- housename: only a LEADING marker junks the value
CREATE OR REPLACE MACRO is_direction_housename(s) AS
    len(tokens(s)) > 0
    AND list_contains(
        ['opp', 'opposite', 'near', 'behind', 'beside',
         'next', 'towards', 'across', 'below'],
        tokens(s)[1]);

-- ---------------- extract.d: isStreetLikeCity ----------------

-- street/premises words in the city column: a city never
-- carries one; "st" stays out, "St Paul's Bay" is real
CREATE OR REPLACE MACRO is_street_like_city(s) AS
    len(list_intersect(tokens(s), [
        'road', 'rd', 'street', 'cross', 'floor',
        'building', 'apartment', 'apt'])) > 0;

-- ---------------- extract.d: isPhone ----------------

-- a contiguous 10+ digit run, or digits with only phone
-- separators and 10+ digits total
CREATE OR REPLACE MACRO is_phone(s) AS
    regexp_matches(s, '\p{N}{10,}')
    OR (NOT regexp_matches(s, '[^\p{N}+.; -]')
        AND len(regexp_replace(s, '[^\p{N}]', '', 'g')) >= 10);

-- ---------------- extract.d: isPostcode ----------------

-- an Indian PIN (6 digits, leading 1-9) or a Maltese code (3
-- letters then 4 digits), spaces packed out.
-- NB: the digits stay ASCII — D digit gates reject unicode
-- digits; only the Maltese letters are \p{L}
CREATE OR REPLACE MACRO is_postcode(s) AS
    regexp_full_match(replace(s, ' ', ''),
                      '[1-9][0-9]{5}|\p{L}{3}[0-9]{4}');

-- ---------------- extract.d: isLevelPack ----------------

-- level packs ("0;1", "ground;-1") are real only when every
-- segment is a level on its own: no empties, no comma crams
CREATE OR REPLACE MACRO is_level_pack(s) AS
    len(list_filter(string_split(s, ';'),
            lambda seg: len(trim(seg)) = 0 OR contains(seg, ','))) = 0;

-- ---------------- extract.d: valueReject ----------------

-- addr:* value data errors, in the D check order: a
-- parenthetical, a phone number, a semicolon multi-value, a
-- comma cram
CREATE OR REPLACE MACRO value_reject(v) AS
    CASE
        WHEN contains(v, '(') OR contains(v, ')') THEN 'parens'
        WHEN is_phone(v) THEN 'phone'
        WHEN contains(v, ';') THEN 'semicolon'
        WHEN contains(v, ',') THEN 'comma'
        ELSE NULL
    END;

-- ---------------- vocab.d: name gates ----------------

-- a cleaned tag value for the buckets: '' when absent
CREATE OR REPLACE MACRO cleaned(tags, key) AS
    strip_edge(clean_invisible(tag(tags, key)));

-- a name value's data error, in the vocab check order: a
-- comma cram, a semicolon multi-value, a parenthetical, a
-- phone number
CREATE OR REPLACE MACRO name_reject(v) AS
    CASE
        WHEN contains(v, ',') THEN 'comma'
        WHEN contains(v, ';') THEN 'semicolon'
        WHEN contains(v, '(') OR contains(v, ')') THEN 'parens'
        WHEN is_phone(v) THEN 'phone'
        ELSE NULL
    END;

-- generic bucket-name filter: 3-40 chars, no underscore, not
-- digits only (D reads Unicode numbers), a letter present,
-- and no name_reject
CREATE OR REPLACE MACRO name_ok(v) AS
    len(v) BETWEEN 3 AND 40
    AND NOT contains(v, '_')
    AND NOT regexp_full_match(v, '[\p{Nd}\p{Nl}\p{No}]+')
    AND regexp_matches(v, '[\p{L}\p{Nl}\p{Mn}\p{Mc}\p{Me}]')
    AND name_reject(v) IS NULL;

-- place value for bucket routing: landuse folds into place
CREATE OR REPLACE MACRO place_of(tags) AS
    CASE WHEN tag(tags, 'place') != '' THEN tag(tags, 'place')
         ELSE tag(tags, 'landuse') END;

-- the locality BIO label of a place value: suburb, quarter
-- and neighbourhood carry their own, everything else is
-- suburb
CREATE OR REPLACE MACRO locality_label(p) AS
    CASE WHEN p IN ('suburb', 'quarter', 'neighbourhood') THEN p
         ELSE 'suburb' END;
