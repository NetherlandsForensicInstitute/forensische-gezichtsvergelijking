from lr_face.utils import cache


def test_cache():
    counter = {"func_with_cache": 0,
               "func_without_cache": 0}

    @cache
    def func_with_cache():
        counter["func_with_cache"] += 1
        return 1

    def func_without_cache():
        counter["func_without_cache"] += 1
        return 1

    func_with_cache()
    assert counter["func_with_cache"] == 1
    func_with_cache()
    assert counter["func_with_cache"] == 1

    func_without_cache()
    assert counter["func_without_cache"] == 1
    func_without_cache()
    assert counter["func_without_cache"] == 2


def test_cache_with_arguments():
    @cache
    def func(a, b):
        return a + b

    func(1, 2)
    assert func.cache_info().hits == 0
    assert func.cache_info().misses == 1
    func(1, 2)
    assert func.cache_info().hits == 1
    assert func.cache_info().misses == 1
    func(a=1, b=2)
    assert func.cache_info().hits == 1
    assert func.cache_info().misses == 2
    func(b=1, a=2)
    assert func.cache_info().hits == 1
    assert func.cache_info().misses == 3
    func(2, 1)
    assert func.cache_info().hits == 1
    assert func.cache_info().misses == 4
    func(1, b=2)
    assert func.cache_info().hits == 1
    assert func.cache_info().misses == 5
    func(1.0, 2.0)  # Floats and ints are treated as the same arguments.
    assert func.cache_info().hits == 2
    assert func.cache_info().misses == 5
