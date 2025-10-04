import duckdb

from lazy_robot_loader.lerobot.internal.query import agg_stats


def test_agg_stats():
    con = duckdb.connect()

    mu, _ = agg_stats("mu", "sigma", "count")

    m = con.query(f"""
    SELECT {mu} AS "mu",
    FROM (
      SELECT
        unnest([10.0, 20.0, 30.0]) AS "mu",
        unnest([1.0, 2.0, 3.0]) AS "sigma",
        unnest([5, 15, 5]) AS "count",
    );
    """).fetch_arrow_table()

    assert m.num_rows == 1
    assert m.num_columns == 1

    assert m.to_pydict()["mu"][0] == 20
