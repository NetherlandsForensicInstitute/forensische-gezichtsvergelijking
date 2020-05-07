from __future__ import annotations

import os
import re
from typing import Optional


class Tag:
    def __init__(self, name: str, version: Optional[int] = None):
        # Convert '-' to '_' for unambiguous filename formats.
        name = name.replace('-', '_')

        if not version and ':' in name:
            name, version = name.split(':')
            version = int(version)

        self.name = name
        self.version = version

    @classmethod
    def from_filename(cls, filename: str) -> Tag:
        basename, _ = os.path.splitext(filename)
        matches = re.search(r'-([^-]+)(?:-(\d+))?$', basename)
        if matches:
            name = matches.group(1)
            version = matches.group(2)
            return cls(name, int(version) if version else None)
        raise ValueError(f'Could not deduce tag from file {filename}')

    @classmethod
    def get_version_from_filename(cls, filename: str) -> Optional[int]:
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
        return f"{basename}-{str(self).replace(':', '-')}{ext}"

    def __str__(self):
        if self.version:
            return f'{self.name}:{self.version}'
        return self.name
