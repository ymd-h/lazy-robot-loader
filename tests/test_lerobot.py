import numpy as np

from lazy_robot_loader.lerobot import LeRobotDataset


def test_pushT(benchmark):
    data = LeRobotDataset(repo_id="lerobot/pusht@v2.1")

    assert len(data) == 25650

    item = data[np.asarray(0)]
    assert isinstance(item, dict)
    assert item["observation.state"].shape == (1, 2)

    def run(i):
        return data[i]

    benchmark(run, np.asarray(0))


def test_pushT_5obs(benchmark):
    data = LeRobotDataset(
        repo_id="lerobot/pusht@v2.1",
        n_observation=5,
    )

    item = data[np.asarray(0)]
    assert isinstance(item, dict)
    assert item["observation.state"].shape == (5, 2)

    def run(i):
        return data[i]

    benchmark(run, np.asarray(0))


def test_pushT_stats():
    v2_0 = LeRobotDataset(repo_id="lerobot/pusht@v2.0").stats
    v2_1 = LeRobotDataset(repo_id="lerobot/pusht@v2.0").stats

    assert len(set(v2_0.keys())) > 0
    assert len(set(v2_0.keys()) ^ set(v2_1.keys())) == 0

    for k in v2_0.keys():
        for m in ("min", "max", "mean", "std"):
            np.testing.assert_allclose(v2_0[k][m], v2_1[k][m])
