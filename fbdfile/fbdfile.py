# fbdfile.py

# Copyright (c) 2012-2025, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Read FLIMbox data and related files (FBD, FBF, FBS.XML).

Fbdfile is a Python library to read FLIMbox data (FBD), firmware (FBF), and
setting (FBS.XML) files. The FLIMbox is an FPGA-based device for high
bandwidth, multi-channel data collection for fluorescence lifetime-resolved
imaging (FLIM) from a pulsed laser scanning confocal microscope.
The files are written by SimFCS and ISS VistaVision software.

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD-3-Clause
:Version: 2025.9.16

Quickstart
----------

Install the fbdfile package and all dependencies from the
`Python Package Index <https://pypi.org/project/fbdfile/>`_::

    python -m pip install -U fbdfile[all]

See `Examples`_ for using the programming interface.

Source code and support are available on
`GitHub <https://github.com/cgohlke/fbdfile>`_.

Requirements
------------

This revision was tested with the following requirements and dependencies
(other versions may work):

- `CPython <https://www.python.org>`_ 3.11.9, 3.12.10, 3.13.7, 3.14.0rc 64-bit
- `NumPy <https://pypi.org/project/numpy>`_ 2.3.3
- `Matplotlib <https://pypi.org/project/matplotlib/>`_ 3.10.6 (optional)
- `Tifffile <https://pypi.org/project/tifffile/>`_ 2025.9.9 (optional)
- `Click <https://pypi.python.org/pypi/click>`_ 8.2.1
  (optional, for command line apps)
- `Cython <https://pypi.org/project/cython/>`_ 3.1.4 (build)

Revisions
---------

2025.9.16

- Initial alpha release based on lfdfiles 2025.7.31.

Notes
-----

The API is not stable yet and might change between revisions.

Python <= 3.10 is no longer supported. 32-bit versions are deprecated.

The latest Microsoft Visual C++ Redistributable for Visual Studio 2015-2022
is required on Windows.

The FLIMbox formats are not documented and might change arbitrarily.
This implementation is based on reverse engineering existing files.
No guarantee can be made as to the correctness of code and documentation.

`SimFCS <https://www.lfd.uci.edu/globals/>`_, a.k.a. Globals for Images,
is software for fluorescence image acquisition, analysis, and simulation,
developed by Enrico Gratton at UCI.

`VistaVision <http://www.iss.com/microscopy/software/vistavision.html>`_
is commercial software for instrument control, data acquisition, and data
processing by ISS Inc (Champaign, IL).

Examples
--------

Read a FLIM lifetime image and metadata from an FBD file:

>>> with FbdFile('tests/data/flimbox$CBCO.fbd') as fbd:
...     bins, times, markers = fbd.decode()
...
>>> print(bins[0, :2], times[:2], markers[:2])
[50 58] [ 0 32] [1944097 2024815]
>>> import numpy
>>> hist = [numpy.bincount(b[b >= 0]) for b in bins]
>>> int(numpy.argmax(hist[0]))
53
>>> image = fbd.asimage((bins, times, markers))
>>> image.shape
(1, 2, 256, 256, 64)

View the histogram and metadata in a FLIMbox data file from the console::

    $ python -m fbdfile tests/data/flimbox$CBCO.fbd

"""

from __future__ import annotations

__version__ = '2025.9.16'

__all__ = [
    '__version__',
    'FbdFile',
    'FbdFileError',
    'fbd_decode',
    'fbd_histogram',
    'fbd_to_b64',
    'fbf_read',
    'fbs_read',
    'sflim_decode',
]

import io
import logging
import os
import re
import struct
import sys
import warnings
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, IO, Literal
    from collections.abc import Sequence, Container

    from numpy.typing import NDArray, ArrayLike

import numpy

from ._fbdfile import fbd_decode, fbd_histogram, sflim_decode


class FbdFileError(Exception):
    """Exception to indicate invalid FLIMbox data file."""


class FbdFile:
    """FLIMbox data file.

    FBD files contain encoded data from the FLIMbox device, storing a
    stream of 16-bit or 32-bit integers (data words) that can be decoded to
    photon arrival windows, channels, and times.

    The measurement's frame size, pixel dwell time, number of sampling
    windows, and scanner type are encoded in the last four characters of the
    file name.
    Newer FBD files, where the 3rd character in the file name tag is `0`,
    start with the first 1kB of the firmware file used for the measurement,
    followed by 31kB containing a binary record with measurement settings.
    FBD files written by VistaVision are accompanied by FBS.XML setting
    files.

    It depends on the application and its settings how to interpret the
    decoded data, for example, as time series, line scans, or image frames
    of FCS or digital frequency domain fluorescence lifetime measurements.

    The data word format depends on the device's firmware.
    A common layout is::

        |F|E|D|C|B|A|9|8|7|6|5|4|3|2|1|0|  data word bits
                            |-----------|  pcc (cross correlation phase)
                        |---------------|  tcc (cross correlation time)
                      |-|                  marker (indicates start of frame)
        |-------------|                    index into decoder table

    The data word can be decoded into a cross correlation phase histogram
    index (shown for the 1st harmonics)::

        bin = (pmax-1 - (pcc + win * (pmax//windows)) % pmax) // pdiv

    - ``bin``, cross correlation phase index (phase histogram bin number).
    - ``pcc``, cross correlation phase (counter).
    - ``pmax``, number of entries in cross correlation phase histogram.
    - ``pdiv``, divisor to reduce number of entries in phase histogram.
    - ``win``, arrival window.
    - ``windows``, number of sampling windows.

    The current implementation uses a decoder table to decode arrival windows
    for each channel. This method is inefficient for 32-bit FLIMbox data
    with large number of windows and channels.

    Parameters:
        filename:
            Name of file to open.
        code:
            Four-character string, encoding frame size (1st char),
            pixel dwell time (2nd char), number of sampling windows
            (3rd char), and scanner type (4th char).
            By default, the code is extracted from the file name.
        frame_size:
            Number of pixels in one line scan, excluding retrace.
        windows:
            Number of sampling windows used by FLIMbox.
        channels:
            Number of channels used by FLIMbox.
        harmonics:
            First or second harmonics.
        pdiv:
            Divisor to reduce number of entries in phase histogram.
        pixel_dwell_time:
            Number of microseconds the scanner remains at each pixel.
        laser_frequency:
            Laser frequency in Hz.
            The default is 20000000 Hz, the internal FLIMbox frequency.
        laser_factor:
            Factor to correct dwell_time/laser_frequency.
            Use when the scanner clock is not known exactly.
        scanner_line_length:
            Number of pixels in each line, including retrace.
        scanner_line_start:
            Index of first valid pixel in scan line.
        scanner_frame_start:
            Index of first valid pixel after marker.
        scanner:
            Scanner software or hardware.
        synthesizer:
            Synthesizer software or hardware.

    """

    _fh: IO[bytes]
    _path: str  # absolute path of file
    _close: bool  # file needs to be closed
    _data_offset: int  # position of raw data in file

    header: numpy.recarray[Any, Any] | None
    """FLIMbox file header, if any."""

    fbf: dict[str, Any] | None
    """FLIMbox firmware header settings, if any."""

    fbs: dict[str, Any] | None
    """FLIMbox settings from FBS.XML file, if any."""

    decoder: str | None
    """Decoder settings function."""

    code: str
    """Four-character string from file name, if any."""

    frame_size: int
    """Number of pixels in one line scan, excluding retrace."""

    windows: int
    """Number of sampling windows used by FLIMbox."""

    channels: int
    """Number of channels used by FLIMbox."""

    harmonics: int
    """First or second harmonics."""

    pdiv: int
    """Divisor to reduce number of entries in phase histogram."""

    pixel_dwell_time: float
    """Number of microseconds the scanner remains at each pixel."""

    laser_frequency: float
    """Laser frequency in Hz."""

    laser_factor: float
    """Factor to correct dwell_time/laser_frequency."""

    scanner_line_length: int
    """Number of pixels in each line, including retrace."""

    scanner_line_start: int
    """Index of first valid pixel in scan line."""

    scanner_frame_start: int
    """Index of first valid pixel after marker."""

    scanner: str
    """Scanner software or hardware."""

    synthesizer: str
    """Synthesizer software or hardware."""

    is_32bit: bool
    """Data words are 32-bit."""

    _attributes = (
        'decoder',
        'frame_size',
        'windows',
        'channels',
        'harmonics',
        'pdiv',
        'pmax',
        'pixel_dwell_time',
        'laser_frequency',
        'laser_factor',
        'synthesizer',
        'scanner',
        'scanner_line_length',
        'scanner_line_start',
        'scanner_frame_start',
    )

    _frame_size: dict[str, int] = {
        # Map 1st character in file name tag to image frame size
        'A': 64,
        'B': 128,
        'C': 256,
        'D': 320,
        'E': 512,
        'F': 640,
        'G': 800,
        'H': 1024,
    }

    _flimbox_settings: dict[str, tuple[int, int, int]] = {
        # Map 3rd character in file name tag to (windows, channels, harmonics)
        # '0': file contains header
        'A': (2, 2, 1),
        'B': (4, 2, 1),
        'C': (8, 2, 1),
        'F': (8, 4, 1),
        'D': (16, 2, 1),
        'E': (32, 2, 1),
        'H': (64, 1, 1),
        # second harmonics. P and Q might be switched in some files?
        'N': (2, 2, 2),
        'O': (4, 2, 2),  # frequency = 40000000 ?
        'P': (8, 2, 2),
        'G': (8, 4, 2),
        'Q': (16, 2, 2),
        'R': (32, 2, 2),
        'S': (64, 1, 2),
    }

    _scanner_settings: dict[str, dict[str, Any]] = {
        # Map 4th and 2nd character in file name tag to pixel_dwell_time,
        # scanner_line_add, scanner_line_start, and scanner_frame_start.
        # As of SimFCS ~2011.
        # These values may not apply any longer and need to be overridden.
        'S': {
            'name': 'Native SimFCS, 3-axis card',
            'A': (4, 198, 99, 0),
            'J': (100, 20, 10, 0),
            'B': (5, 408, 204, 0),
            'K': (128, 16, 8, 0),
            'C': (8, 256, 128, 0),
            'L': (200, 10, 5, 0),
            'D': (10, 204, 102, 0),
            'M': (256, 8, 4, 0),
            'E': (16, 128, 64, 0),
            'N': (500, 4, 2, 0),
            'F': (20, 102, 51, 0),
            'O': (512, 4, 2, 0),
            'G': (32, 64, 32, 0),
            'P': (1000, 2, 1, 0),
            'H': (50, 40, 20, 0),
            'Q': (1024, 2, 1, 0),
            'I': (64, 32, 16, 0),
            'R': (2000, 1, 0, 0),
        },
        'O': {
            'name': 'Olympus FV 1000, NI USB',
            'A': (2, 10, 8, 0),
            'F': (20, 56, 55, 0),
            'B': (4, 10, 8, 0),
            'G': (40, 28, 20, 0),
            'C': (8, 10, 8, 0),
            'H': (50, 12, 10, 0),
            'D': (10, 112, 114, 0),
            'I': (100, 12, 9, 0),
            'E': (12.5, 90, 80, 0),
            'J': (200, 10, 89, 0),
        },
        'Y': {
            'name': 'Zeiss LSM510',
            'A': (6.39, 344, 22, 0),
            'D': (51.21, 176, 22, 414),
            'B': (12.79, 344, 22, 0),
            'E': (102.39, 88, 0, 242),
            'C': (25.61, 344, 22, 0),
            'F': (204.79, 12, 10, 0),
        },
        'Z': {
            'name': 'Zeiss LSM710',
            'A': (6.305, 344, 2, 0),
            'B': (12.79, 344, 2, 0),
            'C': (25.216, 344, 2, 0),
            'D': (50.42, 176, 8, 414),
            'E': (100.85, 88, 10, 242),
            'F': (177.32, 12, 10, 0),
        },
        'I': {
            'name': 'ISS Vista slow scanner',
            'A': (4, 112, 73, 0),
            'H': (32, 37, 28, 0),
            'B': (6, 112, 73, 0),
            'I': (40, 28, 19, 0),
            'C': (8, 112, 73, 0),
            'J': (64, 18, 14, 0),
            'D': (10, 112, 73, 0),
            'K': (200, 6, 4, 0),
            'E': (12.5, 90, 59, 0),
            'L': (500, 6, 4, 0),
            'F': (16, 73, 48, 0),
            'M': (1000, 6, 4, 0),
            'G': (20, 56, 37, 0),
        },
        'V': {
            'name': 'ISS Vista fast scanner',
            'A': (4, 112, 73, 0),
            'H': (32, 37, 28, 0),
            'B': (6, 112, 73, 0),
            'I': (40, 28, 19, 0),
            'C': (8, 112, 73, 0),
            'J': (64, 21, 14, 0),
            'D': (10, 112, 73, 0),
            'K': (100, 12, 8, 0),
            'E': (12.5, 90, 59, 0),
            'L': (200, 6, 4, 0),
            'F': (16, 73, 48, 0),
            'M': (500, 6, 4, 0),
            'G': (20, 56, 37, 0),
            'N': (1000, 6, 4, 0),
        },
        'T': {
            # used with new file format only?
            'name': 'IOTech scanner card'
        },
    }

    _header_t = [
        # Binary header starting at offset 1024 in files with $xx0x names.
        # This is written as a memory dump of a Delphi record, hence the pads
        ('owner', '<i4'),  # must be 0
        ('pixel_dwell_time_index', '<i4'),  # index into SimFCS dropdown ctrl
        ('frame_size_index', '<i4'),  # index into SimFCS dropdown ctrl
        ('line_length', '<i4'),
        ('points_end_of_frame', '<i4'),
        ('x_starting_pixel', '<i4'),
        ('line_integrate', '<i4'),
        ('scanner_index', '<i4'),  # index into SimFCS dropdown ctrl
        ('synthesizer_index', '<i4'),  # index into SimFCS dropdown ctrl
        ('windows_index', '<i4'),  # index into SimFCS dropdown ctrl
        ('channels_index', '<i4'),  # index into SimFCS dropdown ctrl
        ('_pad1', '<i4'),
        ('line_time', '<f8'),
        ('frame_time', '<f8'),
        ('scanner', '<i4'),
        ('_pad2', '<f4'),
        ('laser_frequency', '<f8'),
        ('laser_factor', '<f8'),
        ('frames_to_average', '<i4'),
        ('enabled_before_start', '<i4'),
        # the my?calib fields may be missing
        ('mypcalib', '<16f4'),
        ('mymcalib', '<16f4'),
        ('mypcalib1', '<16f4'),
        ('mymcalib1', '<16f4'),
        ('mypcalib2', '<16f4'),
        ('mymcalib2', '<16f4'),
        ('mypcalib3', '<16f4'),
        ('mymcalib3', '<16f4'),
        ('h1', '<i4'),
        ('h2', '<i4'),
        ('process_enable', 'b'),
        ('integrate', 'b'),
        ('detect_maitai', 'b'),
        ('trigger_on_up', 'b'),
        ('line_scan', 'b'),
        ('circular_scan', 'b'),
        ('average_frames_on_reading', 'b'),
        ('show_each_frame', 'b'),
        ('write_each_frame', 'b'),
        ('write_one_big_file', 'b'),
        ('subtract_background', 'b'),
        ('normalize_to_frames', 'b'),
        ('second_harmonic', 'b'),
        ('_pad3', '3b'),
        ('bin_pixel_by_index', '<i4'),
        ('jitter_level', '<i4'),
        ('phaseshift1', '<i4'),
        ('phaseshift2', '<i4'),
        ('acquire_item_index', '<i4'),
        ('acquire_number_of_frames', '<i4'),
        ('show_channel_index', '<i4'),
    ]

    _header_pixel_dwell_time = (
        # fmt: off
        1, 2, 4, 5, 8, 10, 16, 20, 32, 50, 64,
        100, 128, 200, 256, 500, 512, 1000, 2000,
        # fmt: on
    )

    _header_frame_size = (64, 128, 256, 320, 512, 640, 800, 1024)

    _header_bin_pixel_by = (1, 2, 4, 8)

    _header_windows = (4, 8, 16, 32, 64)

    _header_channels = (
        1,
        2,
        4,
        # the following are not supported
        4,  # 2 to 4 ch
        4,  # 2 ch 8w Spartan6
        4,  # 4 ch 16w frame
    )

    _header_scanners = (
        'IOTech scanner card',
        'Native SimFCS, 3-axis card',
        'Olympus FV 1000, NI USB',
        'Zeiss LSM510',
        'Zeiss LSM710',
        'ISS Vista slow scanner',
        'ISS Vista fast scanner',
        'M2 laptop only',
        'IOTech scanner card',
        'Leica SP8',
        'Zeiss LSM880',
        'Olympus FV3000',
        'Olympus 2P',
    )

    _header_synthesizers = (
        'Internal FLIMbox frequency',
        'Fianium',
        'Spectra Physics MaiTai',
        'Spectra Physics Tsunami',
        'Coherent Chameleon',
    )

    def __init__(
        self,
        file: str | os.PathLike[str] | IO[bytes],
        /,
        *,
        mode: Literal['r', 'r+'] | None = None,
        code: str = '',
        frame_size: int = -1,
        windows: int = -1,
        channels: int = -1,
        harmonics: int = -1,
        pdiv: int = -1,
        pixel_dwell_time: float = -1.0,
        laser_frequency: float = -1.0,
        laser_factor: float = -1.0,
        scanner_line_length: int = -1,
        scanner_line_start: int = -1,
        scanner_frame_start: int = -1,
        scanner: str = '',
        synthesizer: str = '',
    ) -> None:
        """Initialize instance from file name code or file header."""
        if isinstance(file, (str, os.PathLike)):
            if os.path.splitext(file)[1].lower() != '.fbd':
                raise ValueError(f'not a .FBD file: {file!r}')
            if mode is None:
                mode = 'r'
            else:
                if mode[-1:] == 'b':
                    mode = mode[:-1]  # type: ignore[assignment]
                if mode not in {'r', 'r+'}:
                    raise ValueError(f'invalid {mode=!r}')
            self._path = os.path.abspath(file)
            self._close = True
            self._fh = open(self._path, mode + 'b')
        elif hasattr(file, 'seek'):
            self._path = ''
            self._close = False
            self._fh = file
        else:
            raise ValueError(f'cannot open file of type {type(file)}')

        self._fh = self._fh

        self.decoder = None
        self.fbf = None
        self.fbs = None
        self.header = None
        self.code = code
        self.frame_size = frame_size
        self.windows = windows
        self.channels = channels
        self.harmonics = harmonics
        self.pdiv = pdiv
        self.pixel_dwell_time = pixel_dwell_time
        self.laser_frequency = laser_frequency
        self.laser_factor = laser_factor
        self.scanner_line_length = scanner_line_length
        self.scanner_line_start = scanner_line_start
        self.scanner_frame_start = scanner_frame_start
        self.scanner = scanner
        self.synthesizer = synthesizer
        self.is_32bit = False
        self._data_offset = 0

        if not self.code:
            match = re.search(
                r'.*\$([A-Z][A-Z][A-Z0-9][A-Z])\.fbd',
                self.filename,
                re.IGNORECASE,
            )
            if match is None:
                self.code = 'CFCS'  # old FLIMbox file ?
                warnings.warn(
                    'FbdFile: failed to parse code from file name. '
                    f'Using {self.code!r}'
                )
            else:
                self.code = match.group(1)

        assert len(self.code) == 4, code

        if self.code[2].isnumeric() and self._from_fbs():
            pass
        elif self.code[2] == '0':
            self._from_header()
        else:
            self._from_code()

        assert self.windows >= 0
        assert self.channels >= 0
        assert self.harmonics >= 0

        if self.pdiv <= 0:
            try:
                self.pdiv = max(1, self.pmax // 64)
            except AttributeError:
                pass
        assert self.pdiv >= 0

        for attr in self._attributes:
            value = getattr(self, attr)
            if isinstance(value, (int, float)):
                if value < 0:
                    raise ValueError(f"'{attr}' not initialized")
            elif not value:
                # empty str
                raise ValueError(f"'{attr}' not initialized")

    def _from_fbs(self) -> bool:
        """Initialize instance from VistaVision settings file."""
        for ext in ('.FBS.XML', '.fbs.xml'):
            fname = self._path.rsplit('$', 1)[0] + ext
            if os.path.exists(fname):
                break
        else:
            return False
        # self.pdiv = 1
        fbs = fbs_read(fname)
        scn = fbs['ScanParams']
        self.fbf = fbf = fbf_parse_header(fbs['FirmwareParams']['Description'])
        self.is_32bit = '32fifo' in fbf['decoder']
        if self.harmonics < 0:
            self.harmonics = (1, 2)[fbf['secondharmonic']]
        if self.windows < 0:
            self.windows = fbf['windows']
        if self.channels < 0:
            self.channels = fbf['channels']
        if self.synthesizer == '':
            self.synthesizer = 'Unknown'
        if self.scanner == '':
            try:
                self.scanner = scn['ScannerInfo']['ScannerID']
            except IndexError:
                self.scanner = 'Unknown'
        if self.laser_frequency < 0:
            self.laser_frequency = float(scn['ExcitationFrequency'])
        if self.laser_factor < 0:
            self.laser_factor = 1.0
        if self.scanner_line_length < 0:
            self.scanner_line_length = int(scn['ScanLineLength'])
        if self.scanner_line_start < 0:
            self.scanner_line_start = int(scn['ScanLineLeftBorder'])
        if 'PixelOffsetToFrameFlag' in scn:
            # TODO: check this
            self.scanner_frame_start = scn['PixelOffsetToFrameFlag']
        else:
            self.scanner_frame_start = 0
        if self.pixel_dwell_time < 0:
            self.pixel_dwell_time = scn['PixelDwellTime']['value']
            if (
                'Unit' in scn['PixelDwellTime']
                and 'milli' in scn['PixelDwellTime']['Unit'].lower()
            ):
                self.pixel_dwell_time *= 1000
        if self.frame_size < 0:
            self.frame_size = scn['XPixels']
        self.fbs = fbs

        self._fh.seek(0)
        try:
            if self._fh.read(32).decode('cp1252').isprintable():
                # header detected, assume encoded data starts at 33K
                self._data_offset = 33792
        except UnicodeDecodeError:
            pass
        self._fh.seek(0)
        return True

    def _from_code(self) -> None:
        """Initialize instance from file name code."""
        code = self.code
        if self.frame_size < 0:
            self.frame_size = self._frame_size[code[0]]
        if self.windows < 0 or self.channels < 0 or self.harmonics < 0:
            windows, channels, harmonics = self._flimbox_settings[code[2]]
            if self.windows < 0:
                self.windows = windows
            if self.channels < 0:
                self.channels = channels
            if self.harmonics < 0:
                self.harmonics = harmonics
        if (
            self.pixel_dwell_time < 0
            or self.scanner_line_length < 0
            or self.scanner_line_start < 0
            or self.scanner_frame_start < 0
        ):
            (
                pixel_dwell_time,
                scanner_line_add,
                scanner_line_start,
                scanner_frame_start,
            ) = self._scanner_settings[code[3]][code[1]]
            if self.pixel_dwell_time < 0:
                self.pixel_dwell_time = pixel_dwell_time
            if self.scanner_frame_start < 0:
                self.scanner_frame_start = scanner_frame_start
            if self.scanner_line_start < 0:
                self.scanner_line_start = scanner_line_start
            if self.scanner_line_length < 0:
                self.scanner_line_length = self.frame_size + scanner_line_add
        if self.scanner == '':
            self.scanner = self._scanner_settings[code[3]]['name']
        if self.synthesizer == '':
            self.synthesizer = 'Unknown'
        if self.laser_frequency < 0:
            self.laser_frequency = 20000000 * self.harmonics
        if self.laser_factor < 0:
            self.laser_factor = 1.0

    def _from_header(self) -> None:
        """Initialize instance from 32 KB file header."""
        # the first 1024 bytes contain the start of a FLIMbox firmware file
        self.fbf = fbf_read(self._fh.name)
        assert self.fbf is not None
        assert self._fh is not None

        self.is_32bit = '32fifo' in self.fbf['decoder']

        # the next 31kB contain the binary file header
        self._fh.seek(1024)
        self.header = hdr = numpy.fromfile(self._fh, self._header_t, 1)[0]
        if hdr['owner'] != 0:
            raise ValueError(f"unknown header format '{hdr['owner']}'")

        # detect corrupted header: the my?calib fields may be missing
        if (
            hdr['process_enable'] > 1
            or hdr['integrate'] > 1
            or hdr['line_scan'] > 1
            or hdr['subtract_background'] > 1
            or hdr['second_harmonic'] > 1
        ):
            # reload modified header
            self._header_t = FbdFile._header_t[:20] + FbdFile._header_t[28:]
            self._fh.seek(1024)
            self.header = hdr = numpy.fromfile(self._fh, self._header_t, 1)[0]

        if self.harmonics < 0:
            self.harmonics = (1, 2)[self.fbf['secondharmonic']]
        if self.windows < 0:
            windows = self._header_windows[hdr['windows_index']]
            self.windows = self.fbf.get('windows', windows)
            if self.windows != windows:
                logger().warning(
                    'FbdFile: '
                    'windows mismatch between FBF and FBD header '
                    f'({self.windows!r} != {windows!r})'
                )
        if self.channels < 0:
            channels = self._header_channels[hdr['channels_index']]
            self.channels = self.fbf.get('channels', channels)
            if self.channels != channels:
                logger().warning(
                    'FbdFile: '
                    'channels mismatch between FBF and FBD header '
                    f'({self.channels!r} != {channels!r})'
                )
        if self.synthesizer == '':
            try:
                self.synthesizer = self._header_synthesizers[
                    hdr['synthesizer_index']
                ]
            except IndexError:
                self.synthesizer = 'Unknown'
        if self.scanner == '':
            try:
                self.scanner = self._header_scanners[hdr['scanner_index']]
            except IndexError:
                self.scanner = 'Unknown'
        if self.laser_frequency < 0:
            self.laser_frequency = float(hdr['laser_frequency'])
        if self.laser_factor < 0:
            self.laser_factor = float(hdr['laser_factor'])
        if self.scanner_line_length < 0:
            self.scanner_line_length = int(hdr['line_length'])
        if self.scanner_line_start < 0:
            self.scanner_line_start = int(hdr['x_starting_pixel'])
        self.scanner_frame_start = max(self.scanner_frame_start, 0)

        if hdr['frame_time'] >= hdr['line_time'] > 1.0:
            if self.frame_size < 0:
                self.frame_size = round(hdr['frame_time'] / hdr['line_time'])
                for frame_size in self._header_frame_size:
                    if abs(self.frame_size - frame_size) < 3:
                        self.frame_size = frame_size
                        break
            if self.pixel_dwell_time < 0:
                self.pixel_dwell_time = float(
                    hdr['frame_time']
                    / (self.frame_size * self.scanner_line_length)
                )
        else:
            if self.frame_size < 0:
                try:
                    self.frame_size = round(
                        self._header_frame_size[hdr['frame_size_index']]
                    )
                except IndexError:
                    # fall back to filename
                    if self.code:
                        self.frame_size = self._frame_size[self.code[0]]
            if self.pixel_dwell_time < 0:
                try:
                    # use file name
                    dwt = self._scanner_settings[self.code[3]][self.code[1]][0]
                except (IndexError, KeyError, TypeError):
                    try:
                        dwt = self._header_pixel_dwell_time[
                            hdr['pixel_dwell_time_index']
                        ]
                    except IndexError:
                        dwt = 1.0
                self.pixel_dwell_time = dwt

        self._data_offset = 65536  # start of encoded data; contains 2 headers

    @property
    def pmax(self) -> int:
        """Number of entries in cross correlation phase histogram."""
        d = self.decoder_settings
        return int((d['pcc_mask'] >> d['pcc_shr']) + 1) // self.harmonics

    @property
    def scanner_line_add(self) -> int:
        """Number of pixels added to each line (for retrace)."""
        return self.scanner_line_length - self.frame_size

    @property
    def units_per_sample(self) -> float:
        """Number of FLIMbox units per scanner sample."""
        return float(
            (self.pixel_dwell_time * 1e-6)
            * (self.pmax / (self.pmax - 1))
            * (self.laser_frequency * self.laser_factor)
        )

    @cached_property
    def decoder_settings(self) -> dict[str, NDArray[numpy.int16] | int]:
        """Return parameters to decode FLIMbox data stream.

        Returns:
            - 'decoder_table' - Decoder table mapping channel and window
              indices to actual arrival windows.
              The shape is (channels, window indices) and dtype is int16.
            - 'tcc_mask', 'tcc_shr' - Binary mask and number of bits to right
              shift to extract cross correlation time from data word.
            - 'pcc_mask', 'pcc_shr' - Binary mask and number of bits to right
              shift to extract cross correlation phase from data word.
            - 'marker_mask', 'marker_shr' - Binary mask and number of bits to
              right shift to extract markers from data word.
            - 'win_mask', 'win_shr' - Binary mask and number of bits to right
              shift to extract index into lookup table from data word.

        """
        bytes_ = 4 if self.is_32bit else 2
        decoder = f'_b{bytes_}w{self.windows}c{self.channels}'
        if (
            self.windows == 16
            and self.channels == 4
            and self.fbf is not None
            and self.fbf.get('time', '').endswith('Bit')
        ):
            t = self.fbf['time'][:-3]
            decoder += f't{t}'
        self.decoder = decoder
        try:
            settings = getattr(self, decoder)()
        except Exception as exc:
            raise ValueError(f'Decoder{decoder} not implemented') from exc
        return settings  # type: ignore[no-any-return]

    @staticmethod
    def _b4w16c4t10() -> dict[str, NDArray[numpy.int16] | int]:
        # return parameters to decode 32-bit, 16 windows, 4 channels (Spartan6)
        # with line markers
        # 0b00000000000000000000000011111111 cross correlation phase
        # 0b00000000000000000000001111111111 cross correlation time  # 10 bit
        # 0b00000000000000000000010000000000 start of line marker
        # 0b00000000000000000000100000000000 start of frame marker
        # 0b00000000000000000001000000000000 ch0 photon
        # 0b00000000000000011110000000000000 ch0 window
        # 0b00000000000000100000000000000000 ch1 photon
        # 0b00000000001111000000000000000000 ch1 window
        # 0b00000000010000000000000000000000 ch2 photon
        # 0b00000111100000000000000000000000 ch2 window
        # 0b00001000000000000000000000000000 ch3 photon
        # 0b11110000000000000000000000000000 ch3 window
        table = numpy.full((4, 2**20), -1, numpy.int16)
        for i in range(2**20):
            if i & 0b1:
                table[0, i] = (i & 0b11110) >> 1
            if i & 0b100000:
                table[1, i] = (i & 0b1111000000) >> 6
            if i & 0b10000000000:
                table[2, i] = (i & 0b111100000000000) >> 11
            if i & 0b1000000000000000:
                table[3, i] = (i & 0b11110000000000000000) >> 16
        return {
            'decoder_table': table,
            'tcc_mask': 0x3FF,
            'tcc_shr': 0,
            'pcc_mask': 0xFF,
            'pcc_shr': 0,
            # 'line_mask': 0x400,
            # 'line_shr': 10,
            'marker_mask': 0x800,
            'marker_shr': 11,
            'win_mask': 0xFFFFF000,
            'win_shr': 12,
            'swap_words': True,
        }

    @staticmethod
    def _b4w16c4t11() -> dict[str, NDArray[numpy.int16] | int]:
        # return parameters to decode 32-bit, 16 windows, 4 channels (Spartan6)
        # without line markers
        # 0b00000000000000000000000011111111 cross correlation phase
        # 0b00000000000000000000011111111111 cross correlation time  # 11 bit
        # 0b00000000000000000000100000000000 start of frame marker
        # 0b00000000000000000001000000000000 ch0 photon
        # 0b00000000000000011110000000000000 ch0 window
        # 0b00000000000000100000000000000000 ch1 photon
        # 0b00000000001111000000000000000000 ch1 window
        # 0b00000000010000000000000000000000 ch2 photon
        # 0b00000111100000000000000000000000 ch2 window
        # 0b00001000000000000000000000000000 ch3 photon
        # 0b11110000000000000000000000000000 ch3 window
        return {
            **FbdFile._b4w16c4t10(),
            'tcc_mask': 0x7FF,
            'marker_mask': 0x800,
            'marker_shr': 11,
        }

    @staticmethod
    def _b4w8c4() -> dict[str, NDArray[numpy.int16] | int]:
        # return parameters to decode 32-bit, 8 windows, 4 channels (Spartan-6)
        # 0b00000000000000000000000000000001 ch0 photon
        # 0b00000000000000000000000000001110 ch0 window
        # 0b00000000000000000000000000010000 ch1 photon
        # 0b00000000000000000000000011100000 ch1 window
        # 0b00000000000000000000000100000000 ch2 photon
        # 0b00000000000000000000111000000000 ch2 window
        # 0b00000000000000000001000000000000 ch3 photon
        # 0b00000000000000001110000000000000 ch3 window
        # 0b00000000111111110000000000000000 cross correlation phase
        # 0b00011111111111110000000000000000 cross correlation time
        # 0b01000000000000000000000000000000 start of frame marker
        table = numpy.full((4, 2**16), -1, numpy.int16)
        for i in range(2**16):
            if i & 0b1:
                table[0, i] = (i & 0b1110) >> 1
            if i & 0b10000:
                table[1, i] = (i & 0b11100000) >> 5
            if i & 0b100000000:
                table[2, i] = (i & 0b111000000000) >> 9
            if i & 0b1000000000000:
                table[3, i] = (i & 0b1110000000000000) >> 13
        return {
            'decoder_table': table,
            'tcc_mask': 0x1FFF0000,
            'tcc_shr': 16,
            'pcc_mask': 0xFF0000,
            'pcc_shr': 16,
            'marker_mask': 0x40000000,
            'marker_shr': 30,
            'win_mask': 0xFFFF,
            'win_shr': 0,
            # 'swap_words': True,  # ?
        }

    @staticmethod
    def _b2w4c2() -> dict[str, NDArray[numpy.int16] | int]:
        # return parameters to decode 4 windows, 2 channels
        # fmt: off
        table = numpy.array(
            [[-1, 0, -1, 1, -1, 2, -1, 3, -1, 0, -1, -1, 1, 0, -1, 2,
              1, 0, 3, 2, 1, 0, 3, 2, 1, -1, 3, 2, -1, -1, 3, -1],
             [-1, -1, 0, -1, 1, -1, 2, -1, 3, 0, -1, -1, 0, 3, -1, 0,
              1, 2, 2, 1, 2, 1, 1, 2, 3, -1, 0, 3, -1, -1, 3, -1]],
            numpy.int16
        )
        # fmt: on
        return {
            'decoder_table': table,
            'tcc_mask': 0x3FF,
            'tcc_shr': 0,
            'pcc_mask': 0xFF,
            'pcc_shr': 0,
            'marker_mask': 0x400,
            'marker_shr': 10,
            'win_mask': 0xF800,
            'win_shr': 11,
        }

    @staticmethod
    def _b2w8c2() -> dict[str, NDArray[numpy.int16] | int]:
        # return parameters to decode 8 windows, 2 channels
        # TODO: this does not correctly decode data acquired with ISS firmware
        table = numpy.full((2, 81), -1, numpy.int16)
        table[0, 1:9] = range(8)
        table[1, 9:17] = range(8)
        table[:, 17:] = numpy.mgrid[0:8, 0:8].reshape(2, -1)[::-1, :]
        return {
            'decoder_table': table,
            'tcc_mask': 0xFF,
            'tcc_shr': 0,
            'pcc_mask': 0x3F,
            'pcc_shr': 0,
            'marker_mask': 0x100,
            'marker_shr': 8,
            'win_mask': 0xFFFF,
            'win_shr': 9,
        }

    @staticmethod
    def _b2w8c4() -> dict[str, NDArray[numpy.int16] | int]:
        # return parameters to decode 8 windows, 4 channels
        logger().warning(
            'FbdFile: b2w8c4 decoder not tested. '
            'Please submit an FBD file to https://github.com/cgohlke/fbdfile'
        )
        table = numpy.full((4, 128), -1, numpy.int16)
        for i in range(128):
            win = i & 0b0000111
            ch0 = (i & 0b0001000) >> 3
            ch1 = (i & 0b0010000) >> 4
            ch2 = (i & 0b0100000) >> 5
            ch3 = (i & 0b1000000) >> 6
            if ch0 + ch1 + ch2 + ch3 != 1:
                continue
            if ch0:
                table[0, i] = win
            elif ch1:
                table[1, i] = win
            elif ch2:
                table[2, i] = win
            elif ch3:
                table[3, i] = win
        return {
            'decoder_table': table,
            'tcc_mask': 0xFF,
            'tcc_shr': 0,
            'pcc_mask': 0x3F,
            'pcc_shr': 0,
            'marker_mask': 0x100,
            'marker_shr': 8,
            'win_mask': 0xFFFF,
            'win_shr': 9,
        }

    @staticmethod
    def _b2w16c1() -> dict[str, NDArray[numpy.int16] | int]:
        # return parameters to decode 16 windows, 1 channel
        logger().warning(
            'FbdFile: b2w16c1 decoder not tested. '
            'Please submit an FBD file to https://github.com/cgohlke/fbdfile'
        )
        table = numpy.full((1, 32), -1, numpy.int16)
        for i in range(32):
            win = (i & 0b11110) >> 1
            ch0 = i & 0b00001
            if ch0:
                table[0, i] = win
        return {
            'decoder_table': table,
            'tcc_mask': 0xFF,
            'tcc_shr': 0,
            'pcc_mask': 0x3F,
            'pcc_shr': 0,
            'marker_mask': 0x100,
            'marker_shr': 8,
            'win_mask': 0xFFFF,
            'win_shr': 11,
        }

    @staticmethod
    def _b2w16c2() -> dict[str, NDArray[numpy.int16] | int]:
        # return parameters to decode 16 windows, 2 channels
        logger().warning(
            'FbdFile: b2w16c2 decoder not tested. '
            'Please submit an FBD file to https://github.com/cgohlke/fbdfile'
        )
        table = numpy.full((2, 64), -1, numpy.int16)
        for i in range(64):
            win = (i & 0b111100) >> 2
            ch0 = (i & 0b000010) >> 1
            ch1 = i & 0b000001
            if ch0 + ch1 != 1:
                continue
            if ch0:
                table[0, i] = win
            elif ch1:
                table[1, i] = win
        return {
            'decoder_table': table,
            'tcc_mask': 0xFF,
            'tcc_shr': 0,
            'pcc_mask': 0x3F,
            'pcc_shr': 0,
            'marker_mask': 0x100,
            'marker_shr': 8,
            'win_mask': 0xFFFF,
            'win_shr': 10,
        }

    @staticmethod
    def _b2w32c2() -> dict[str, NDArray[numpy.int16] | int]:
        # return parameters to decode 32 windows, 2 channels
        # TODO
        raise NotImplementedError(
            'FbdFile: b2w32c2 decoder not implemented. '
            'Please submit an FBD file to https://github.com/cgohlke/fbdfile'
        )

    @staticmethod
    def _b2w64c1() -> dict[str, NDArray[numpy.int16] | int]:
        # return parameters to decode 64 windows, 1 channel
        # TODO
        raise NotImplementedError(
            'FbdFile: b2w64c1 decoder not implemented. '
            'Please submit an FBD file to https://github.com/cgohlke/fbdfile'
        )

    def decode(
        self,
        data: NDArray[Any] | None = None,
        *,
        word_count: int = -1,
        skip_words: int = 0,
        max_markers: int = 65536,
        **kwargs: Any,
    ) -> tuple[
        NDArray[numpy.int8 | numpy.int16],
        NDArray[numpy.uint32 | numpy.uint64],
        NDArray[numpy.intp],
    ]:
        """Return decoded records from FLIMbox data stream.

        Parameters:
            data:
                FLIMbox data stream. By default, the data is read from file.
            word_count:
                Number of data words to process.
                By default, all words are processed.
            skip_words:
                Number of data words to skip at beginning of stream.
            max_markers:
                Maximum number of markers expected in data stream.

        Returns:
            bins:
                Cross correlation phase index for all channels and data points.
                A int8 array of shape (channels, size).
                A value of -1 means no photon was counted.
            times:
                The times in FLIMbox counter units at each data point.
                An array of type uint64 or uint32, potentially huge.
            markers:
                The indices of up markers in the data stream, usually
                indicating frame starts. An array of type ssize_t.

        """
        assert self._fh is not None
        dtype = numpy.dtype('<u4' if self.is_32bit else '<u2')
        if data is None:
            self._fh.seek(self._data_offset + skip_words * dtype.itemsize, 0)
            try:
                data = numpy.fromfile(self._fh, dtype=dtype, count=word_count)
            except io.UnsupportedOperation:
                data = numpy.frombuffer(
                    self._fh.read(dtype.itemsize * word_count), dtype=dtype
                )
        elif skip_words or word_count != -1:
            if word_count < 0:
                data = data[skip_words:word_count]
            else:
                data = data[skip_words : skip_words + word_count]
        if data.dtype != dtype:
            raise ValueError('invalid data dtype')

        bins_out = numpy.empty(
            (self.channels, data.size),
            dtype=numpy.int8 if self.pmax // self.pdiv <= 128 else numpy.int16,
        )
        times_out = numpy.empty(data.size, dtype=numpy.uint64)
        markers_out = numpy.zeros(max_markers, dtype=numpy.intp)

        fbd_decode(
            data,
            bins_out,
            times_out,
            markers_out,
            self.windows,
            self.pdiv,
            self.harmonics,
            **self.decoder_settings,
        )

        markers_out = markers_out[markers_out > 0]
        if len(markers_out) == max_markers:
            warnings.warn(
                f'number of markers exceeded buffer size {max_markers}'
            )

        return bins_out, times_out, markers_out

    def frames(
        self,
        records: tuple[NDArray[Any], NDArray[Any], NDArray[Any]] | None = None,
        /,
        *,
        select_frames: slice | None = None,
        aspect_range: tuple[float, float] = (0.8, 1.2),
        frame_cluster: int = 0,
        **kwargs: Any,
    ) -> tuple[tuple[int, int], list[tuple[int, int]]]:
        """Return shape and start/stop indices of scanner frames.

        If unable to detect any frames using the default settings, try to
        determine a correction factor from clusters of frame durations.

        Parameters:
            records:
                Bins, times, and markers from decode function.
                By default, call :py:meth:`FbdFile.decode`.
            select_frames:
                Specifies which image frames to return.
                By default, all frames are returned.
            aspect_range:
                Minimum and maximum aspect ratios of valid frames.
                The default lets 1:1 aspect pass.
            frame_cluster:
                Index of the frame duration cluster to use when calculating
                the correction factor.
            **kwargs:
                Additional arguments passed to :py:meth:`FbdFile.decode`.

        Returns:
            1. shape: Dimensions of scanner frame.
            2. frame_markers: Start and stop indices of detected image frames.

        """
        if records is None:
            records = self.decode(**kwargs)
        times, markers = records[-2:]

        line_time = self.scanner_line_length * self.units_per_sample
        frame_durations = numpy.ediff1d(times[markers])

        frame_markers = []
        if aspect_range:
            # detect frame markers assuming correct settings
            line_num = sys.maxsize
            for i, duration in enumerate(frame_durations):
                lines = duration / line_time
                aspect = self.frame_size / lines
                if aspect_range[0] < aspect < aspect_range[1]:
                    frame_markers.append(
                        (int(markers[i]), int(markers[i + 1]) - 1)
                    )
                else:
                    continue
                line_num = min(line_num, lines)
            line_num = int(round(line_num))

        if not frame_markers:
            # calculate frame duration clusters, assuming few clusters that
            # are narrower and more separated than cluster_size.
            cluster_size = 1024
            clusters: list[list[int]] = []
            cluster_indices = []
            for d in frame_durations:
                d = int(d)
                for i, c in enumerate(clusters):
                    if abs(d - c[0]) < cluster_size:
                        cluster_indices.append(i)
                        c[0] = min(c[0], d)
                        c[1] += 1
                        break
                else:
                    cluster_indices.append(len(clusters))
                    clusters.append([d, 1])
            clusters = list(sorted(clusters, key=lambda x: x[1], reverse=True))
            # possible correction factors, assuming square frame shape
            line_num = self.frame_size
            laser_factors = [c[0] / (line_time * line_num) for c in clusters]
            # select specified frame cluster
            frame_cluster = min(frame_cluster, len(laser_factors) - 1)
            self.laser_factor = laser_factors[frame_cluster]
            frame_markers = [
                (int(markers[i]), int(markers[i + 1]) - 1)
                for i, c in enumerate(cluster_indices)
                if c == frame_cluster
            ]
            msg = [
                'no frames detected with default settings. '
                'Using square shape and correction factor '
                f'{self.laser_factor:.5f}.'
            ]
            if len(laser_factors) > 1:
                factors = ', '.join(f'{i:.5f}' for i in laser_factors[:4])
                msg.append(
                    f'The most probable correction factors are: {factors}'
                )
            warnings.warn('\n'.join(msg))

        if not isinstance(select_frames, slice):
            select_frames = slice(select_frames)
        frame_markers = frame_markers[select_frames]
        if not frame_markers:
            raise ValueError('no frames selected')
        return (line_num, self.scanner_line_length), frame_markers

    def asimage(
        self,
        records: tuple[NDArray[Any], NDArray[Any], NDArray[Any]] | None = None,
        frames: tuple[tuple[int, int], list[tuple[int, int]]] | None = None,
        /,
        *,
        integrate_frames: int = 1,
        square_frame: bool = True,
        **kwargs: Any,
    ) -> NDArray[numpy.uint16]:
        """Return image histograms from decoded records and detected frames.

        This function may fail to produce expected results when settings
        were recorded incorrectly, scanner and FLIMbox frequencies were out
        of sync, or scanner settings were changed during acquisition.

        Parameters:
            records:
                Bins, times, and markers from decode function.
                By default, call :py:meth:`FbdFile.decode`.
            frames:
                Scanner_shape and frame_markers from frames function.
                By default, call :py:meth:`FbdFile.frames`.
            integrate_frames:
                Specifies which frames to sum. By default, all frames are
                summed into one. If 0, no frames are summed.
            square_frame:
                If True, return square image (frame_size x frame_size),
                else return full scanner frame.
            **kwargs:
                Additional arguments passed to :py:meth:`FbdFile.decode` and
                :py:meth:`FbdFile.frames`.

        Returns:
            Image histogram of shape (number of frames, channels in bins
            array, detected line numbers, frame_size, histogram bins).

        """
        kwargs_frames = parse_kwargs(
            kwargs, 'select_frames', 'aspect_range', 'frame_cluster'
        )
        if records is None:
            records = self.decode(**kwargs)
        bins, times, markers = records
        if frames is None:
            frames = self.frames(records, **kwargs_frames)
        scanner_shape, frame_markers = frames
        # an extra line to scanner frame to allow clipping indices
        scanner_shape = scanner_shape[0] + 1, scanner_shape[1]
        # allocate output array of scanner frame shape
        shape = (
            integrate_frames if integrate_frames else len(frame_markers),
            bins.shape[0],  # channels
            scanner_shape[0] * scanner_shape[1],
            self.pmax // self.pdiv,
        )
        result = numpy.zeros(shape, dtype=numpy.uint16)
        # calculate frame data histogram
        fbd_histogram(
            bins,
            times,
            frame_markers,
            self.units_per_sample,
            self.scanner_frame_start,
            result,
        )
        # reshape frames and slice valid region
        result = result.reshape(shape[:2] + scanner_shape + shape[-1:])
        if square_frame:
            result = result[
                ...,
                : self.frame_size,
                self.scanner_line_start : self.scanner_line_start
                + self.frame_size,
                :,
            ]
        return result

    @property
    def filehandle(self) -> IO[bytes]:
        """File handle."""
        return self._fh

    @property
    def filename(self) -> str:
        """Name of file or empty if binary stream."""
        return os.path.basename(self._path)

    @property
    def dirname(self) -> str:
        """Directory containing file."""
        return os.path.dirname(self._path)

    @property
    def name(self) -> str:
        """Name of file or handle."""
        if self._path:
            return os.path.basename(self._path)
        if hasattr(self._fh, 'name') and self._fh.name:
            return self._fh.name
        return ''

    def close(self) -> None:
        """Close file handle and free resources."""
        if self._close:
            try:
                self._fh.close()
            except Exception:
                pass

    def __enter__(self) -> FbdFile:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        name = 'BytesIO' if isinstance(self._fh, io.BytesIO) else self.name
        return f'<{self.__class__.__name__} {name!r}>'

    def __str__(self) -> str:
        info = [repr(self)]
        info.extend(f'{a}: {getattr(self, a)}' for a in self._attributes)
        if self.fbf is not None:
            info.append(
                indent(
                    'firmware:',
                    *(f'{k}: {v}'[:64] for k, v in self.fbf.items()),
                )
            )
        if self.header is not None:
            info.append(
                indent(
                    'header:',
                    *(
                        f'{k}: {self.header[k]}'[:64]
                        for k, _ in self._header_t
                        if k[0] != '_'
                    ),
                )
            )
        info.append(
            indent(
                'decoder_settings:',
                *(
                    (
                        f'{k}: {v:#x}'
                        if 'mask' in k
                        else f'{k}: {v}'[:64].splitlines()[0]
                    )
                    for k, v in self.decoder_settings.items()
                ),
            )
        )
        return indent(*info)

    def plot(self, *, show: bool = True) -> None:
        """Plot histogram and image for all channels."""
        from matplotlib import pyplot
        from tifffile import imshow

        assert self.pmax is not None
        assert self.pdiv is not None

        fig = pyplot.figure(facecolor='w')
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(self.name)
        ax.set_xlabel('Bin')
        ax.set_ylabel('Counts')
        ax.set_xlim((0, self.pmax // self.pdiv - 1))
        bins, times, markers = self.decode()
        bins_channel: Any  # for mypy
        for ch, bins_channel in enumerate(bins):
            histogram = numpy.bincount(bins_channel[bins_channel >= 0])
            ax.plot(histogram, label=f'Ch{ch}')
        ax.legend()
        pyplot.tight_layout()

        image = self.asimage((bins, times, markers))
        if image.sum(dtype=numpy.uint32) > 0:
            img = image.sum(axis=(0, -1), dtype=numpy.uint32)
            imshow(
                img,
                title=self.name,
                photometric='minisblack',
                interpolation='nearest',
                show=False,
            )

        if show:
            pyplot.show()


def fbs_read(file: str | os.PathLike[str] | IO[str], /) -> dict[str, Any]:
    """Return metadata from FLIMbox settings (FBS.XML) file.

    VistaVision FBS.XML files contain FLIMbox acquisition settings in XML
    format.

    Parameters:
        filename: Name of file to open.

    Examples:
        >>> fbs = fbs_read('tests/data/flimbox.fbs.xml')
        >>> fbs['ScanParams']['ExcitationFrequency']
        20000000

    """
    fh: IO[str]
    if isinstance(file, (str, os.PathLike)):
        fh = open(file, encoding='utf-8')
        close = True
    elif hasattr(file, 'seek'):
        fh = file
        close = False
    else:
        raise ValueError(f'cannot open file of type {type(file)}')

    try:
        buf = fh.read(100)
        if not ('<?xml ' in buf and '<FastFlimFbdDataSettings>' in buf):
            raise ValueError('invalid FBS file format')
        fh.seek(0)
        buf = fh.read()
    finally:
        if close:
            try:
                fh.close()
            except Exception:
                pass

    info = xml2dict(buf)
    result: dict[str, Any] = info['FastFlimFbdDataSettings']
    if not result:
        raise ValueError('no FastFlimFbdDataSettings element found')
    return result


def fbf_read(
    file: str | os.PathLike[str] | IO[bytes],
    /,
    *,
    firmware: bool = False,
    maxheaderlength: int = 1024,
) -> dict[str, Any]:
    """Return metadata from FLIMbox device firmware (FBF) file.

    FBF files contain FLIMbox device firmwares, stored in binary form
    following a NULL terminated ASCII string containing properties and
    description.

    Parameters:
        filename:
            Name of FBF file to open.
        firmware:
            Include the firmware binary data.
        maxheaderlength:
            Maximum length of ASCII header.

    Examples:
        >>> header = fbf_read('tests/data/flimbox.fbf')
        >>> header['windows']
        16
        >>> header['channels']
        2
        >>> header['secondharmonic']
        0
        >>> 'extclk' in header
        True

    """
    fh: IO[bytes]
    if isinstance(file, (str, os.PathLike)):
        fh = open(file, 'rb')
        close = True
    elif hasattr(file, 'seek'):
        fh = file
        close = False
    else:
        raise ValueError(f'cannot open file of type {type(file)}')

    try:
        try:
            # the first 1024 bytes contain the header
            buffer = fh.read(maxheaderlength).split(b'\x00', 1)[0]
        except ValueError as exc:
            raise ValueError('invalid FBF header') from exc
        header = bytes2str(buffer)
        meta = fbf_parse_header(header)
        if not meta:
            raise ValueError('failed to parse FBF header')
        if firmware:
            if len(header) != maxheaderlength:
                fh.seek(len(header) + 1)
            meta['firmware'] = fh.read()
    finally:
        if close:
            try:
                fh.close()
            except Exception:
                pass
    return meta


def fbf_parse_header(header: str, /) -> dict[str, Any]:
    """Return FLIMbox firmware settings from header."""
    settings: dict[str, Any] = {}
    try:
        header, comment_ = header.rsplit('/', 1)
        comment = [comment_]
    except ValueError:
        comment = []
    for name, value, unit in re.findall(
        r'([a-zA-Z\s]*)([.\d]*)([a-zA-Z\d]*)/', header
    ):
        name = name.strip().lower()
        if not name:
            name = {'w': 'windows', 'ch': 'channels'}.get(unit, None)
            unit = ''
        if not name:
            comment.append(name + value + unit)
            continue
        if unit == 'MHz':
            unit = 1000000
        try:
            if unit:
                value = int(value) * int(unit)
            else:
                value = int(value)
            unit = 0
        except ValueError:
            pass
        settings[name] = (value + unit) if value != '' else True
    cstr = '/'.join(reversed(comment))
    if cstr:
        settings['comment'] = cstr
    return settings


def _fbd_decode(
    data: Any,
    bins_out: Any,
    times_out: Any,
    markers_out: Any,
    windows: Any,
    pdiv: Any,
    harmonics: Any,
    decoder_table: Any,
    tcc_mask: Any,
    tcc_shr: Any,
    pcc_mask: Any,
    pcc_shr: Any,
    marker_mask: Any,
    marker_shr: Any,
    win_mask: Any,
    win_shr: Any,
) -> None:
    """Decode FLIMbox data stream.

    See the documentation of the FbdFile class for parameter descriptions
    and the _fbdfile.pyx file for a faster implementation.

    This implementation is for reference only. Do not use!

    """
    tcc = data & tcc_mask  # cross correlation time
    if tcc_shr:
        tcc >>= tcc_shr
    times_out[:] = tcc
    times_out[times_out == 0] = (tcc_mask >> tcc_shr) + 1
    times_out[0] = 0
    times_out[1:] -= tcc[:-1]
    del tcc
    numpy.cumsum(times_out, out=times_out)

    markers = data & marker_mask
    markers = numpy.diff(markers.view(numpy.int16))
    markers = numpy.where(markers > 0)[0]  # trigger up
    markers += 1
    size = min(len(markers), len(markers_out))
    markers_out[:size] = markers[:size]
    del markers

    if win_mask != 0xFFFF:  # window index
        win = data & win_mask
        win >>= win_shr
    else:
        win = data >> win_shr
    win = decoder_table.take(win, axis=1)
    nophoton = win == -1
    pmax = (pcc_mask >> pcc_shr + 1) // harmonics
    win *= pmax // windows * harmonics
    pcc = data & pcc_mask  # cross correlation phase
    if pcc_shr:
        pcc >>= pcc_shr
    win += pcc
    del pcc
    win %= pmax
    win += 1 - pmax
    numpy.negative(win, win)
    if pdiv > 1:
        win //= pdiv
    win = win.astype(numpy.int8)
    win[nophoton] = -1
    bins_out[:] = win


def _fbd_histogram(
    bins: Any,
    times: Any,
    frame_markers: Any,
    units_per_sample: Any,
    scanner_frame_start: Any,
    out: Any,
) -> None:
    """Calculate histograms from decoded FLIMbox data and frame markers.

    See the documentation of the FbdFile class for parameter descriptions
    and the _fbdfile.pyx file for a much faster implementation.

    This implementation is for reference only. Do not use!

    """
    nframes, nchannels, frame_length, nwindows = out.shape
    for f, (j, k) in enumerate(frame_markers):
        f = f % nframes
        t = times[j:k] - times[j]
        t /= units_per_sample  # index into flattened array
        if scanner_frame_start:
            t -= scanner_frame_start
        t = t.astype(numpy.uint32, copy=False)
        numpy.clip(t, 0, frame_length - 1, out=t)
        for c in range(nchannels):
            d = bins[c, j:k]
            for w in range(nwindows):
                x = numpy.where(d == w)[0]
                x = t.take(x)
                x = numpy.bincount(x, minlength=frame_length)
                out[f, c, :, w] += x


def fbd_to_b64(
    fbdfile: str | os.PathLike[Any],
    /,
    b64files: str = '{filename}_c{channel:02}t{frame:04}.b64',
    *,
    integrate_frames: int = 0,
    square_frame: bool = True,
    pdiv: int = -1,
    laser_frequency: float = -1,
    laser_factor: float = -1.0,
    pixel_dwell_time: float = -1.0,
    frame_size: int = -1,
    scanner_line_length: int = -1,
    scanner_line_start: int = -1,
    scanner_frame_start: int = -1,
    cmap: str = 'turbo',
    verbose: bool = True,
    show: bool = True,
) -> None:
    """Convert SimFCS FLIMbox data file to B64 files.

    Parameters:
        fbdfile:
            FLIMbox data file to convert.
        b64files:
            Format string for B64 output file names.
        integrate_frames:
            Specifies which frames to sum. By default, no frames are summed.
        square_frame:
            If True, return square image (frame_size x frame_size).
            Else return full scanner frame.
            Only works if `show` is true.
        pdiv:
            Divisor to reduce number of entries in phase histogram.
        laser_frequency:
            Laser frequency in Hz.
        laser_factor:
            Factor to correct dwell_time/laser_frequency.
        pixel_dwell_time:
            Number of microseconds the scanner remains at each pixel.
        frame_size:
            Number of pixels in one line scan, excluding retrace.
        scanner_line_length:
            Number of pixels in each line, including retrace.
        scanner_line_start:
            Index of first valid pixel in scan line.
        scanner_frame_start:
            Index of first valid pixel after marker.
        cmap:
            Matplotlib colormap for plotting images.
        verbose:
            Print detailed information about file and conversion.
        show:
            If True, plot (but do not write) the decoded frames.
            If False, one B64 files is written for each frame and channel.

    """
    prints: Any = print if verbose else nullfunc

    with FbdFile(
        fbdfile,
        pdiv=pdiv,
        laser_frequency=laser_frequency,
        laser_factor=laser_factor,
        pixel_dwell_time=pixel_dwell_time,
        frame_size=frame_size,
        scanner_line_length=scanner_line_length,
        scanner_line_start=scanner_line_start,
        scanner_frame_start=scanner_frame_start,
    ) as fbd:
        prints(fbd)
        records = fbd.decode()
        bins, times, markers = records
        # prints(records)
        frames = fbd.frames(records)
        # prints(frames)
        image = fbd.asimage(
            records,
            frames,
            integrate_frames=integrate_frames,
            square_frame=square_frame,
        )
        # prints(image)
        image = numpy.moveaxis(image, -1, 2)
        times_ = times / (fbd.laser_frequency * fbd.laser_factor)
        if show:
            from matplotlib import pyplot
            from tifffile import imshow

            pyplot.figure()
            pyplot.title('Bins and Markers')
            pyplot.xlabel('time (s)')
            pyplot.ylabel('Cross correlation phase index')
            pyplot.plot(times_, bins[0], '.', alpha=0.5, label='Ch0')
            pyplot.vlines(
                x=times_[markers],
                ymin=-1,
                ymax=1,
                colors='red',
                label='Markers',
            )
            pyplot.legend()

            pyplot.figure()
            pyplot.title('Cross correlation phase histogram')
            pyplot.xlabel('Bin')
            pyplot.ylabel('Counts')
            bins_channel: Any  # for mypy
            for ch, bins_channel in enumerate(records[0]):
                histogram = numpy.bincount(bins_channel[bins_channel >= 0])
                pyplot.plot(histogram, label=f'Ch{ch}')
            pyplot.legend()

            if not integrate_frames and image.shape[0] > 1:
                image = numpy.moveaxis(image, 0, 1)
                title = 'Photons[Channel, Frame, Bin, Y, X]'
            else:
                title = 'Photons[Channel, Bin, Y, X]'
            imshow(
                image,
                vmax=None,
                photometric='minisblack',
                cmap=cmap,
                title=title,
            )
            pyplot.show()
        else:
            prints()
            for j in range(image.shape[0]):
                for i in range(image.shape[1]):
                    b64name = b64files.format(
                        filename=fbdfile, channel=i, frame=j
                    )
                    prints(b64name, flush=True)
                    b64_write(
                        b64name, image[j, i].astype(numpy.int16, copy=False)
                    )


def b64_write(
    filename: os.PathLike[Any] | str,
    data: ArrayLike,
    /,
) -> None:
    """Write array of square int16 images to B64 file."""
    data = numpy.asarray(data)
    if data.dtype.char != 'h':
        raise ValueError(f'invalid data type {data.dtype} (must be int16)')
    # TODO: write carpet
    if data.ndim != 3 or data.shape[1] != data.shape[2]:
        raise ValueError(f'invalid shape {data.shape}')
    with open(filename, 'wb') as fh:
        fh.write(struct.pack('I', data.shape[-1]))
        data.tofile(fh)


def parse_kwargs(
    kwargs: dict[str, Any],
    /,
    *keys: str,
    _del: bool = True,
    **keyvalues: Any,
) -> dict[str, Any]:
    """Return dict with keys from keys|keyvals and values from kwargs|keyvals.

    Examples:
        >>> kwargs = {'one': 1, 'two': 2, 'four': 4}
        >>> kwargs2 = parse_kwargs(kwargs, 'two', 'three', four=None, five=5)
        >>> kwargs == {'one': 1}
        True
        >>> kwargs2 == {'two': 2, 'four': 4, 'five': 5}
        True

    """
    result = {}
    for key in keys:
        if key in kwargs:
            result[key] = kwargs[key]
            if _del:
                del kwargs[key]
    for key, value in keyvalues.items():
        if key in kwargs:
            result[key] = kwargs[key]
            if _del:
                del kwargs[key]
        else:
            result[key] = value
    return result


def format_dict(
    adict: dict[str, Any],
    /,
    *,
    prefix: str = '',
    indent: str = '  ',
    bullets: tuple[str, str] = ('', ''),
    excludes: Sequence[str] = ('_',),
    linelen: int = 79,
    trim: int = 0,
) -> str:
    """Return pretty-print of nested dictionary."""
    result = []
    for k, v in sorted(adict.items(), key=lambda x: str(x[0]).lower()):
        if any(k.startswith(e) for e in excludes):
            continue
        if isinstance(v, dict):
            v = '\n' + format_dict(
                v, prefix=prefix + indent, excludes=excludes, trim=0
            )
            result.append(f'{prefix}{bullets[1]}{k}: {v}')
        else:
            result.append((f'{prefix}{bullets[0]}{k}: {v}')[:linelen].rstrip())
    if trim > 0:
        result[0] = result[0][trim:]
    return '\n'.join(result)


def nullfunc(*args: Any, **kwargs: Any) -> None:
    """Null function."""
    return


def stripnull(string: bytes) -> bytes:
    r"""Return byte string truncated at first null character.

    Use to clean NULL terminated C strings.

    >>> stripnull(b'bytes\x00\x00b')
    b'bytes'

    """
    i = string.find(b'\x00')
    return string if i < 0 else string[:i]


def bytes2str(
    b: bytes, /, encoding: str | None = None, errors: str = 'strict'
) -> str:
    """Return Unicode string from encoded bytes up to first NULL character."""
    if encoding is None or '16' not in encoding:
        i = b.find(b'\x00')
        if i >= 0:
            b = b[:i]
    else:
        # utf-16
        i = b.find(b'\x00\x00')
        if i >= 0:
            b = b[: i + i % 2]

    try:
        return b.decode('utf-8' if encoding is None else encoding, errors)
    except UnicodeDecodeError:
        if encoding is not None:
            raise
        return b.decode('cp1252', errors)


def xml2dict(
    xml: str,
    /,
    *,
    sanitize: bool = True,
    prefix: tuple[str, str] | None = None,
    exclude: Container[str] | None = None,
    sep: str = ',',
) -> dict[str, Any]:
    """Return XML as dictionary.

    Parameters:
        xml: XML data to convert.
        sanitize: Remove prefix from etree Element.
        prefix: Prefixes for dictionary keys.
        exclude: Ignore element tags.
        sep: Sequence separator.

    Returns:
        dict: Dictionary representation of XML element.

    """
    from collections import defaultdict
    from xml.etree import ElementTree

    at, tx = prefix if prefix else ('', '')
    exclude = set() if exclude is None else exclude

    def astype(value: Any, /) -> Any:
        # return string value as int, float, bool, tuple, or unchanged
        if not isinstance(value, str):
            return value
        if sep and sep in value:
            # sequence of numbers?
            values = []
            for val in value.split(sep):
                v = astype(val)
                if isinstance(v, str):
                    return value
                values.append(v)
            return tuple(values)
        for t in (int, float, asbool):
            try:
                return t(value)
            except (TypeError, ValueError):
                pass
        return value

    def etree2dict(t: ElementTree.Element, /) -> dict[str, Any] | None:
        # adapted from https://stackoverflow.com/a/10077069/453463
        key = t.tag
        if sanitize:
            key = key.rsplit('}', 1)[-1]
        if key in exclude:
            return None
        d: dict[str, Any] = {key: {} if t.attrib else None}
        children = list(t)
        if children:
            dd = defaultdict(list)
            for dc in map(etree2dict, children):
                if dc is not None:
                    for k, v in dc.items():
                        dd[k].append(astype(v))
            d = {
                key: {
                    k: astype(v[0]) if len(v) == 1 else astype(v)
                    for k, v in dd.items()
                }
            }
        if t.attrib:
            d[key].update((at + k, astype(v)) for k, v in t.attrib.items())
        if t.text:
            text = t.text.strip()
            if children or t.attrib:
                if text:
                    d[key][tx + 'value'] = astype(text)
            else:
                d[key] = astype(text)
        return d

    result = etree2dict(ElementTree.fromstring(xml))
    return {} if result is None else result


def asbool(
    value: str,
    /,
    true: Sequence[str] | None = None,
    false: Sequence[str] | None = None,
) -> bool | bytes:
    """Return string as boolean if possible, else raise TypeError.

    >>> asbool('ON', ['on'], ['off'])
    True

    """
    value = value.strip().lower()
    if true is None:
        if value == 'true':
            return True
    elif value in true:
        return True
    if false is None:
        if value == 'false':
            return False
    elif value in false:
        return False
    raise TypeError


def indent(*args: Any) -> str:
    """Return joined string representations of objects with indented lines."""
    text = '\n'.join(str(arg) for arg in args)
    return '\n'.join(
        ('  ' + line if line else line) for line in text.splitlines() if line
    )[2:]


def logger() -> logging.Logger:
    """Return logger for liffile module."""
    return logging.getLogger('liffile')


def askopenfilename(**kwargs: Any) -> str:
    """Return file name(s) from Tkinter's file open dialog."""
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    root.update()
    filenames = filedialog.askopenfilename(**kwargs)
    root.destroy()
    return filenames


def main() -> int:
    """Command line usage main function."""
    import click

    @click.group()
    @click.version_option(version=__version__)
    def cli() -> None:
        pass

    # @cli.command(help='Convert files to TIFF.')
    # @click.option(
    #     '--format',
    #     default='tiff',
    #     help='Output file format.',
    #     type=click.Choice(['tiff']),
    # )
    # @click.option(
    #     '--compress',
    #     default=0,
    #     help='Zlib compression level.',
    #     type=click.IntRange(0, 10, clamp=False),
    # )
    # @click.argument('files', nargs=-1, type=click.Path(dir_okay=False))
    # def convert(format: str, compress: int, files: Any) -> None:
    #     if not files:
    #         files = askopenfilename(
    #             title='Select FBD file(s)',
    #             multiple=True,
    #             filetypes=[('FBD files', '*.FBD')],
    #         )
    #     if files:
    #         convert2tiff(files, compress=compress)

    @cli.command(help='View data in file.')
    @click.argument('files', nargs=-1, type=click.Path(dir_okay=False))
    def view(files: Any) -> None:
        if not files:
            files = askopenfilename(
                title='Select FLIMbox data file(s)',
                filetypes=[('FBD files', '*.FBD')],
            )
        if files:
            if isinstance(files, (list, tuple)):
                files = files[0]
            with FbdFile(files) as fbd:
                print(fbd)
                fbd.plot()

    if len(sys.argv) == 1:
        sys.argv.append('view')
    elif len(sys.argv) == 2 and os.path.isfile(sys.argv[1]):
        sys.argv.append(sys.argv[1])
        sys.argv[1] = 'view'

    cli(prog_name='fbdfile')
    return 0


if __name__ == '__main__':
    sys.exit(main())
