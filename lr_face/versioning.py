from __future__ import annotations

import os
import re


class Tag:
    def __init__(self, name: str, version: int = 0):
        if ':' in name and not version:
            name, version = name.split(':')
            version = int(version)

        # Convert '-' to '_' for unambiguous filename formats.
        self.name = name.replace('-', '_')
        self.version = version

    @classmethod
    def from_filename(cls, filename: str) -> Tag:
        basename, _ = os.path.splitext(filename)
        matches = re.search(r'-([^-]+)-(\d+)$', basename)
        if matches:
            name, version = matches.groups()
            return cls(name, int(version))
        raise ValueError(f'Could not deduce tag from file {filename}')

    @classmethod
    def get_version_from_filename(cls, filename: str) -> int:
        return cls.from_filename(filename).version

    def append_to_filename(self, filename: str) -> str:
        """
        Takes a filename and appends this tag to it.

        Example:

        ```python
        Tag('tag', 1).append_to_filename('test.txt')  # 'test-tag-1.txt'
        ```

        :param filename: str
        :return: str
        """
        basename, ext = os.path.splitext(filename)
        return f'{basename}-{self.name}-{self.version}{ext}'

    def __str__(self):
        return f'{self.name}:{self.version}'
