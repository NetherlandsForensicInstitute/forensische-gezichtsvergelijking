from collections import defaultdict
from typing import Tuple, Dict

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


def test_cache_with_class_methods():
    class CacheTest:
        """
        Dummy class that can multiply two numbers and add a bias term. It keeps
        track of the number of times `multiply` has been computed (not called!)
        with a certain set of arguments. Since we cache the method, the counter
        should initially be 0 and then be increased to 1 the first time it is
        called. Subsequent calls should not increment the counter any further.
        """

        def __init__(self, bias):
            self.counter: Dict[Tuple[float, float], int] = defaultdict(int)
            self.bias = bias

        @cache
        def multiply(self, a: float, b: float):
            self.counter[(a, b)] += 1
            return a * b + self.bias

        def __hash__(self):
            return hash(self.bias)

        def __eq__(self, other):
            return self.bias == other.bias

    # Create 3 instances of this dummy class.
    # The first is unique in that it has a `bias` of 1.
    instance1 = CacheTest(1)

    # The second has a `bias` of 2, so it's different from `instance1`.
    instance2 = CacheTest(2)

    # The third also has a `bias` of 2, so its hash is the same as `instance2`.
    instance3 = CacheTest(2)

    # When we call `multiply` on the first instance we expect the counter to go
    # from 0 to 1 for `instance1`, but remain 0 for `instance2`.
    assert instance1.counter[(2, 3)] == 0
    assert instance2.counter[(2, 3)] == 0
    instance1.multiply(2, 3)
    assert instance1.multiply.cache_info().hits == 0
    assert instance1.counter[(2, 3)] == 1
    assert instance2.counter[(2, 3)] == 0

    # When we call `multiply` a second time, we should see no further changes
    # in the counter.
    instance1.multiply(2, 3)
    assert instance1.counter[(2, 3)] == 1
    assert instance2.counter[(2, 3)] == 0

    # We should see a cache hit. The cache hits are apparently stored at the
    # class-level, so all other instances of the same class will always have
    # the same number of cache hits. This behavior is weird and so we should
    # preferably not depend on this, since it's bound to introduce errors.
    assert instance1.multiply.cache_info().hits == 1
    assert instance2.multiply.cache_info().hits == 1
    assert instance3.multiply.cache_info().hits == 1

    # When we now call `multiply` on `instance2`, we expect its counter to be
    # increased to 1, and subsequent calls should not affect it.
    assert instance2.counter[(2, 3)] == 0
    instance2.multiply(2, 3)
    assert instance2.counter[(2, 3)] == 1
    instance2.multiply(2, 3)
    assert instance2.counter[(2, 3)] == 1

    # Now, since `instance2` and `instance3` have the same hash, we expect that
    # when we call `instance3.multiply` with the same arguments with which we
    # previously called `instance2.multiply` the cached result is returned,
    # thus the counter of `instance3` should NOT be incremented and remain 0.
    assert instance3.counter[(2, 3)] == 0
    instance3.multiply(2, 3)
    assert instance3.counter[(2, 3)] == 0
