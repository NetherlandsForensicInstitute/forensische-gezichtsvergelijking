from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class Version:
    major: int = 0
    minor: int = 0
    micro: int = 0

    @classmethod
    def from_filename(cls, filename: str) -> Version:
        matches = re.search(r'_(\d(?:\.\d+)+)\.\w+$', filename)
        if matches:
            return cls.from_string(matches.group(1))
        raise ValueError(f'Could not deduce version from filename {filename}')

    @classmethod
    def from_string(cls, string: str) -> Version:
        matches = re.search(r'^(\d+)(?:\.(\d+))?(?:\.(\d+))?$', string)
        if matches:
            major, minor, micro = (int(d or 0) for d in matches.groups())
            return cls(major, minor, micro)
        raise ValueError(f'Could not deduce version from string {string}')

    @property
    def suffix(self) -> str:
        return f'_{str(self)}'

    def append_to_filename(self, filename: str) -> str:
        basename, ext = os.path.splitext(filename)
        return f'{basename}{self.suffix}{ext}'

    def increment(self, major: bool = False, minor: bool = False) -> Version:
        if major and minor:
            raise ValueError(
                'Cannot increment major and minor version simultaneously')
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
