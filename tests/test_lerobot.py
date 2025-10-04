import numpy as np

from lazy_robot_loader.lerobot import LeRobotDataset


def test_pushT():
    data = LeRobotDataset(repo_id="lerobot/pusht@v2.1")

    assert len(data) == 25650

    item = data[np.asarray(0)]
    assert isinstance(item, dict)
    assert item["observation.state"].shape == (1, 2)

    assert set(item.keys()) == (
        {
            "observation.state",
            "observation.image",
            "observation_is_pad",
            "action",
            "action_is_pad",
            "episode_index",
            "frame_index",
            "timestamp",
            "index",
            "task_index",
        }
    )

    np.testing.assert_allclose(
        item["observation.state"],
        np.asarray([[222.0, 97.0]]),
    )
    assert item["observation.image"].shape == (1, 96, 96, 3)
    np.testing.assert_allclose(
        item["observation_is_pad"],
        np.asarray([False]),
    )
    np.testing.assert_allclose(
        item["action"],
        np.asarray(
            [
                [
                    233.0,
                    71.0,
                ]
            ]
        ),
    )
    np.testing.assert_allclose(
        item["action_is_pad"],
        np.asarray([False]),
    )
    np.testing.assert_allclose(
        item["episode_index"],
        np.asarray([0]),
    )
    np.testing.assert_allclose(
        item["frame_index"],
        np.asarray([0]),
    )
    np.testing.assert_allclose(
        item["timestamp"],
        np.asarray([0.0]),
    )
    np.testing.assert_allclose(
        item["index"],
        np.asarray([0]),
    )
    np.testing.assert_allclose(
        item["task_index"],
        np.asarray([0]),
    )


def test_pushT_5obs():
    data = LeRobotDataset(
        repo_id="lerobot/pusht@v2.1",
        n_observation=5,
    )

    item = data[np.asarray(0)]
    assert isinstance(item, dict)
    assert item["observation.state"].shape == (5, 2)


def test_pushT_episode_boundary():
    data = LeRobotDataset(
        repo_id="lerobot/pusht@v2.0",
        n_action=2,
    )

    item = data[160]
    np.testing.assert_allclose(
        item["action_is_pad"],
        np.asarray([False, True]),
    )


def test_pushT_stats():
    v2_0 = LeRobotDataset(repo_id="lerobot/pusht@v2.0").stats
    v2_1 = LeRobotDataset(repo_id="lerobot/pusht@v2.1").stats

    assert len(set(v2_0.keys())) > 0
    assert set(v2_0.keys()) == set(v2_1.keys())

    for k in v2_0.keys():
        for m in ("min", "max", "mean", "std"):
            np.testing.assert_allclose(
                getattr(v2_0[k], m),
                getattr(v2_1[k], m),
                rtol=1e-3,
                atol=1e-3,
            )
            assert getattr(v2_0[k], m).ndim in (1, 3)
