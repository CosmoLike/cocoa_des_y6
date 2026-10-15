# Histogram reconstruction study

The source/lens reconstruction options are available in every likelihood YAML:
`photoz_histogram_akima: 0` keeps the original midpoint curve; value `1` selects
Akima after fixed `photoz_histogram_akima_factor: 3`, `5` or `7` preparation.
Both paths always use uniform-grid lookup. The first factor never changes
with numerical accuracy.

![Original, three-, five- and seven-point reconstructions](../nz_akima_preparation.png)

DES Y6 is excluded from the statistical Δχ² study because its current supplied
identity metric is not a physical survey covariance. See the other seven
projects' `tests/studies/README.md` for measured distances. The former expensive
reconstruction comparisons are no longer collected by ordinary pytest.
The frozen likelihood regressions and shared synthetic reader checks remain.
