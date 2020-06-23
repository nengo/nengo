from nengo.utils.ipython import check_ipy_version


def test_check_ipy_version():
    assert check_ipy_version((1, 2))
    assert not check_ipy_version((999, 999))
