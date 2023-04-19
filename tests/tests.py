import pytest
import fespy


def test_lon_lat_to_cartesian():
    assert fespy.lon_lat_to_cartesian(0, 0) == (6371000.0, 0.0, 0.0)
    assert fespy.lon_lat_to_cartesian(30, 0) == pytest.approx(
        (
            5517447.847510659,
            3185499.9999999995,
            0.0,
        )
    )
