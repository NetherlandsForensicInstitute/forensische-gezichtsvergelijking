from __future__ import annotations

import os
import re
from typing import Union, Iterator


class Version:
    def __init__(self, major: Union[str, int], minor: int = 0, micro: int = 0):
        """
        Instantiate a `Version` instance in one of two ways:
            (1) pass along a full version string as the only argument, e.g.
                `Version('0.2.6')`, the other two arguments are then ignored;
            (2) pass along three separate integers for the major, minor and
                micro versions, e.g. `Version(0, 2, 6)`.

        :param major: Union[str, int]
        :param minor: int
        :param micro: int
        """
        if isinstance(major, str):
            self.major, self.minor, self.micro = self.from_string(major)
        else:
            self.major = major
            self.minor = minor
            self.micro = micro

    @classmethod
    def from_filename(cls, filename: str) -> Version:
        """
        Takes a filename and extracts the version from it. It assumes the
        version is at the end of the filename, just before the extension, and
        separated from the rest of the filename by an underscore.

        Example:
        ```python
        Version.from_filename('test_0.2.6.txt')  # Version('0.2.6')
        ```

        :param filename: str
        :return: Version
        """
        matches = re.search(r'_(\d(?:\.\d+)+)\.\w+$', filename)
        if matches:
            return cls.from_string(matches.group(1))
        raise ValueError(f'Could not deduce version from filename {filename}')

    @classmethod
    def from_string(cls, string: str) -> Version:
        """
        Takes a version (e.g. "0.2.6") and turns it into a `Version` instance.

        :param string: str
        :return: Version
        """
        matches = re.search(r'^(\d+)(?:\.(\d+))?(?:\.(\d+))?$', string)
        if matches:
            major, minor, micro = (int(d or 0) for d in matches.groups())
            return cls(major, minor, micro)
        raise ValueError(f'Could not deduce version from string {string}')

    def append_to_filename(self, filename: str) -> str:
        """
        Takes a filename and adds the version number to it.

        Example:

        ```python
        Version('0.0.1').append_to_filename('test.txt')  # 'test_0.0.1.txt'
        ```

        :param filename: str
        :return: str
        """
        basename, ext = os.path.splitext(filename)
        return f'{basename}_{str(self)}{ext}'

    def increment(self, major: bool = False, minor: bool = False) -> Version:
        """
        Returns a new `Version` instance with its micro version bumped by 1. If
        `major` or `minor` is True, the major or minor version is bumped,
        respectively, instead.

        Examples:

        ```python
         Version("0.0.1").increment()  # Version("0.0.2")
         Version("0.1.4").increment()  # Version("0.1.5")
         Version("1.3.6").increment()  # Version("1.3.7")
         Version("0.0.1").increment(minor=True)  # Version("0.1.0")
         Version("0.1.4").increment(minor=True)  # Version("0.2.0")
         Version("1.3.6").increment(minor=True)  # Version("1.4.0")
         Version("1.3.6").increment(major=True)  # Version("2.0.0")
         ```
        """
        if major and minor:
            raise ValueError('Cannot increment both major and minor version')
        if major:
            return Version(self.major + 1, 0, 0)
        if minor:
            return Version(self.major, self.minor + 1, 0)
        return Version(self.major, self.minor, self.micro + 1)

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: Union[str, Version]) -> bool:
        if isinstance(other, str):
            other = Version.from_string(other)
        return isinstance(other, self.__class__) \
               and self.major == other.major \
               and self.minor == other.minor \
               and self.micro == other.micro

    def __gt__(self, other: Union[str, Version]):
        if isinstance(other, str):
            other = self.from_string(other)
        return self.major > other.major \
               or (self.major == other.major and self.minor > other.minor) \
               or (self.major == other.major
                   and self.minor == other.minor
                   and self.micro > other.micro)

    def __ge__(self, other: Union[str, Version]):
        return self > other or self == other

    def __lt__(self, other: Union[str, Version]):
        return not (self > other or self == other)

    def __le__(self, other: Union[str, Version]):
        return self < other or self == other

    def __str__(self) -> str:
        return f'{self.major}.{self.minor}.{self.micro}'

    def __iter__(self) -> Iterator[int]:
        return iter([self.major, self.minor, self.micro])
