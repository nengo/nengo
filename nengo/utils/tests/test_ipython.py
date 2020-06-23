from nengo.utils.ipython import check_ipy_version


def test_check_ipy_version():
    assert check_ipy_version((1, 2)) is True
    assert check_ipy_version((999, 999)) is False
