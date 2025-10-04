import duckdb
import numpy as np

from lazy_robot_loader.lerobot.internal.query import agg_vector, agg_stats
from lazy_robot_loader.lerobot.internal.functional import to_array


def test_agg_vector():
    con = duckdb.connect()

    m = agg_vector("max", "a", 3)

    mm = to_array(
        con.query(f"""
    SELECT {m} AS "max",
    FROM (
      SELECT
        unnest([[1.0, 2.0, 3.0], [4.0, 5.0, 7.0], [-1.2, 9.2, 3.4]]) AS "a"
    );
    """).fetch_arrow_table()["max"]
    )

    assert mm.shape == (1, 3)
    np.testing.assert_allclose(mm, np.asarray([[4.0, 9.2, 7.0]]))


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
