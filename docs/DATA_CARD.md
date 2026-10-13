# Data card: open UEFA Champions League match results

## Provenance

| | |
|---|---|
| Sources | [footballcsv/europe-champions-league](https://github.com/footballcsv/europe-champions-league) (seasons 1992/93-2010/11) and [openfootball/champions-league](https://github.com/openfootball/champions-league) (2011/12-2025/26) |
| Licence | **CC0 1.0 Universal** (public-domain dedication) for both; the data may be copied, modified and redistributed without restriction |
| Access | Anonymous HTTPS download from `raw.githubusercontent.com`. No login, API key or Kaggle token |
| Pinned versions | footballcsv `66393c8dc5c86b459dc700a0ae3a0bd0abedb56e` (2020-04-12), openfootball `abfaeddc2ee3d14f99ecc163c9ddb46cb4d67cef` (2026-07-02, "auto-update week 27") |
| First retrieved | 2026-10-12 |
| Integrity | SHA-256 of all 34 downloaded files in `src/ucl/checksums.json`, verified on every `make data` |
| Redistribution here | Raw files and the parsed table are *not* committed (they are rebuilt in seconds by `make data`) |

The raw files total ~0.6 MB. The openfootball repository carries the later seasons only (it does not
contain the 1990s), and footballcsv stops in 2016, hence the combination. The two overlap on 2011/12-2014/15,
which is used as a cross-check (below).

## Contents after parsing (`data/processed/matches.csv`)

4,127 matches, 34 seasons (16 Sep 1992 - 30 May 2026), 180 clubs, 42 national associations.

| column | meaning |
|---|---|
| `season` | start year (2022 = 2022/23) |
| `date` | calendar date of the match |
| `stage`, `round`, `group`, `leg` | `group`, `league` (2024+), `playoff`, `knockout`, `final`; round label; group letter when the source provides it; leg 1/2 for two-legged ties |
| `home`, `away`, `*_country` | canonical club key such as `real madrid|ESP`, and the association |
| `home_goals`, `away_goals` | **90-minute** score (the modelling target) |
| `et_*`, `pen_*` | score after extra time and penalty shoot-out, when played |

Matches per season: 61-85 in the 1990s group-stage era, 157 in 1999-2002 (two group stages), 125 for the
standard 32-team format, 119 in 2019/20 (single-leg knockouts in Lisbon) and 189 for the 36-team league phase
of 2024/25 and 2025/26.

## Processing decisions

* **Qualifying rounds are dropped** (round names starting `Qual.`, `Prelim.`, `Playoff Round`; stage `Qualifying`): hundreds of tiny clubs with a handful of matches each.
* **Club names** are harmonised by normalisation (ASCII-fold, drop `FC`/`CF`/...) plus a 25-entry hand-built alias table (`src/ucl/names.py`), e.g. `Bayern München` / `FC Bayern München`, `Olympique Lyonnais` / `Olympique Lyon`, `Monaco (MCO)` / `Monaco (FRA)`.
* **Legs** are recovered for openfootball (which does not label them): a knockout pairing that occurs twice in a round is a two-legged tie.
* **Neutral venues**: finals, and the quarter-finals onward of 2019/20 (Lisbon). Home/away order in a final is arbitrary in the sources.

## Quality checks (automated in `tests/test_data.py` and notebook 01)

* Exactly one final per season; champion derived from the final's extra-time/penalty result is correct for all 34 editions.
* Match counts per season equal the known tournament formats.
* **Cross-source validation**: on the 500 matches of 2011/12-2014/15 that both sources contain, 497 agree exactly on date, teams and 90-minute score; the other 3 are finals listed with the teams in opposite order (mirrored scores).
* No missing scores or dates; no club plays itself.

## Known limitations

* No venue, attendance, line-ups, injuries, referee or odds; no domestic-league information.
* 2020/21 was played largely behind closed doors; some 2020/21 and 2019/20 fixtures were moved to neutral venues. This is not in the data.
* A name variant that the alias table misses would split one club into two entities (the alias table was reviewed by listing every (country, normalised-name) pair).
* Source repositories are community-maintained; individual score errors cannot be ruled out beyond the cross-check above.
