# test_fbdfile.py

# Copyright (c) 2012-2025, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Unittests for the fbdfile package.

:Version: 2025.9.18

"""

import glob
import io
import os
import pathlib
import sys
import sysconfig

import numpy
import pytest
from lfdfiles import SimfcsB64
from numpy.testing import assert_almost_equal, assert_array_equal

import fbdfile
from fbdfile import (
    FbdFile,
    FbdFileError,
    __version__,
    fbd_decode,
    fbd_histogram,
    fbd_to_b64,
    fbf_read,
    fbs_read,
    sflim_decode,
)
from fbdfile._fbdfile import sflim_decode_photons
from fbdfile.fbdfile import xml2dict

HERE = pathlib.Path(os.path.dirname(__file__))
DATA = HERE / 'data'
SHOW = False

try:
    from matplotlib import pyplot
    from tifffile import imshow
except ImportError:
    imshow = None  # type: ignore[assignment]
    SHOW = False

try:
    import lfdfiles
except ImportError:
    lfdfiles = None  # type: ignore[assignment]

try:
    import phasorpy
except ImportError:
    phasorpy = None  # type: ignore[assignment]


@pytest.mark.skipif(__doc__ is None, reason='__doc__ is None')
def test_version():
    """Assert fbdfile versions match docstrings."""
    ver = ':Version: ' + __version__
    assert ver in __doc__
    assert ver in fbdfile.__doc__


def test_fbd_error():
    """Test FbdFile errors."""
    fname = DATA / 'flimbox.fbf'
    if not os.path.exists(fname):
        pytest.skip(f'{fname!r} not found')

    with pytest.raises(FileNotFoundError):
        FbdFile('nonexistingfile.fbd')
    with pytest.raises(ValueError):
        FbdFile(fname)


def test_fbd_cbco_b2w8c2():
    """Test read CBCO b2w8c2 FBD file."""
    # SimFCS 16-bit with code settings; does not correctly decode image
    fname = DATA / 'flimbox$CBCO.fbd'
    if not os.path.exists(fname):
        pytest.skip(f'{fname!r} not found')

    with FbdFile(fname) as fbd:
        assert str(fbd).startswith("<FbdFile 'flimbox$CBCO.fbd'>")
        assert fbd.filename == os.path.basename(fname)
        assert fbd.dirname == os.path.dirname(fname)
        assert fbd.name == fbd.filename
        assert fbd.filehandle
        assert fbd.header is None
        assert fbd.fbf is None
        assert fbd.fbs is None
        assert fbd.decoder == '_b2w8c2'
        assert fbd.code == 'CBCO'
        assert fbd.frame_size == 256
        assert fbd.windows == 8
        assert fbd.channels == 2
        assert fbd.harmonics == 1
        assert fbd.pdiv == 1
        assert fbd.pixel_dwell_time == 4.0
        assert fbd.laser_frequency == 20000000.0
        assert fbd.laser_factor == 1.0
        assert fbd.scanner_line_length == 266
        assert fbd.scanner_line_start == 8
        assert fbd.scanner_frame_start == 0
        assert fbd.scanner == 'Olympus FV 1000, NI USB'
        assert fbd.synthesizer == 'Unknown'
        assert not fbd.is_32bit

        bins, times, markers = fbd.decode(
            word_count=500000, skip_words=1900000
        )

        assert bins.shape == (2, 500000)
        assert bins.dtype == numpy.int8
        assert bins[0, :2].tolist() == [53, 51]
        hist = [numpy.bincount(b[b >= 0]) for b in bins]
        assert numpy.argmax(hist[0]) == 53

        assert times.shape == (500000,)
        assert times.dtype == numpy.uint64
        assert times[:2].tolist() == [0, 42]

        assert markers.shape == (2,)
        assert markers.dtype == numpy.int64
        assert markers.tolist() == [44097, 124815]

        with pytest.warns(UserWarning):
            shape, frame_markers = fbd.frames(
                (bins, times, markers),
                select_frames=slice(None),
                aspect_range=(0.8, 1.2),
                frame_cluster=0,
            )

        assert shape == (256, 266)
        assert frame_markers.tolist() == [[44097, 124814]]

        image = fbd.asimage(
            (bins, times, markers),
            (shape, frame_markers),
            integrate_frames=1,
            square_frame=True,
        )
        assert image.shape == (1, 2, 256, 256, 64)
        assert image.dtype == numpy.uint16
        assert image[0, 0, 128, 128].sum() == 2
        counts = image.sum(axis=(0, 2, 3, 4), dtype=numpy.uint32)
        assert counts.tolist() == [72976, 0]

        if SHOW:
            imshow(image[:, 0].sum(-1, dtype=numpy.uint32), show=True)


def test_fbd_cfco_b2w8c2():
    """Test read CFCO b2w8c2 FBD file."""
    # SimFCS 16-bit with correct code settings
    fname = DATA / 'CeruleanVenusCell1$CFCO.fbd'
    if not os.path.exists(fname):
        pytest.skip(f'{fname!r} not found')

    with FbdFile(fname) as fbd:
        assert str(fbd).startswith("<FbdFile 'CeruleanVenusCell1$CFCO.fbd'>")
        assert fbd.filename == os.path.basename(fname)
        assert fbd.dirname == os.path.dirname(fname)
        assert fbd.name == fbd.filename
        assert fbd.filehandle
        assert fbd.header is None
        assert fbd.fbf is None
        assert fbd.fbs is None
        assert fbd.decoder == '_b2w8c2'
        assert fbd.code == 'CFCO'
        assert fbd.frame_size == 256
        assert fbd.windows == 8
        assert fbd.channels == 2
        assert fbd.harmonics == 1
        assert fbd.pdiv == 1
        assert fbd.pixel_dwell_time == 20.0
        assert fbd.laser_frequency == 20000000.0
        assert fbd.laser_factor == 1.0
        assert fbd.scanner_line_length == 312
        assert fbd.scanner_line_start == 55
        assert fbd.scanner_frame_start == 0
        assert fbd.scanner == 'Olympus FV 1000, NI USB'
        assert fbd.synthesizer == 'Unknown'
        assert not fbd.is_32bit

        bins, times, markers = fbd.decode()

        assert bins.shape == (2, 10506240)
        assert bins.dtype == numpy.int8
        assert bins[0, :2].tolist() == [26, -1]
        hist = [numpy.bincount(b[b >= 0]) for b in bins]
        assert numpy.argmax(hist[0]) == 47

        assert times.shape == (10506240,)
        assert times.dtype == numpy.uint64
        assert times[:2].tolist() == [0, 115]

        assert markers.shape == (50,)
        assert markers.dtype == numpy.int64
        assert markers[[0, -1]].tolist() == [185694, 10299672]

        shape, frame_markers = fbd.frames(
            (bins, times, markers),
            select_frames=slice(None),
            aspect_range=(0.8, 1.2),
            frame_cluster=0,
        )

        assert shape == (256, 312)
        assert frame_markers[0].tolist() == [192126, 529435]

        image = fbd.asimage(
            (bins, times, markers),
            (shape, frame_markers),
            integrate_frames=0,
            square_frame=True,
        )
        assert image.shape == (29, 2, 256, 256, 64)
        assert image.dtype == numpy.uint16
        assert image[0, 0, 128, 128].sum() == 7
        counts = image.sum(axis=(0, 2, 3, 4), dtype=numpy.uint32)
        assert counts.tolist() == [2973148, 2642822]

        if SHOW:
            imshow(image.sum(-1, dtype=numpy.uint32), show=True)


def test_fbd_xx2x_b4w16c4t10():
    """Test read XX2X b4w16c4t10 FBD file."""
    # ISS VistaVision 32-bit with external fbs.xml settings
    fname = DATA / 'IFLItest/303 Cu6 vs FL$XX2X.fbd'
    if not os.path.exists(fname):
        pytest.skip(f'{fname!r} not found')

    with FbdFile(fname) as fbd:
        assert str(fbd).startswith("<FbdFile '303 Cu6 vs FL$XX2X.fbd'>")
        assert fbd.filename == os.path.basename(fname)
        assert fbd.dirname == os.path.dirname(fname)
        assert fbd.name == fbd.filename
        assert fbd.filehandle
        assert fbd.header is None
        assert fbd.fbf['channels'] == 4
        assert fbd.fbs['ScanParams']['Channels'] == 1
        assert fbd.decoder == '_b4w16c4t10'
        assert fbd.code == 'XX2X'
        assert fbd.frame_size == 256
        assert fbd.windows == 16
        assert fbd.channels == 4
        assert fbd.harmonics == 1
        assert fbd.pdiv == 4
        assert fbd.pixel_dwell_time == 32.0
        assert fbd.laser_frequency == 20000000.0
        assert fbd.laser_factor == 1.0
        assert fbd.scanner_line_length == 291
        assert fbd.scanner_line_start == 24
        assert fbd.scanner_frame_start == 0
        assert fbd.scanner == 'ISS Scanning Mirror V2'
        assert fbd.synthesizer == 'Unknown'
        assert fbd.is_32bit

        bins, times, markers = fbd.decode()

        assert bins.shape == (4, 2650112)
        assert bins.dtype == numpy.int8
        assert bins[0, 1000000:1000002].tolist() == [-1, 47]
        hist = [numpy.bincount(b[b >= 0]) for b in bins]
        assert numpy.argmax(hist[0]) == 49

        assert times.shape == (2650112,)
        assert times.dtype == numpy.uint64
        assert times[:2].tolist() == [0, 1024]

        assert markers.shape == (39,)
        assert markers.dtype == numpy.int64
        assert markers[[0, -1]].tolist() == [4753, 2635056]

        shape, frame_markers = fbd.frames(
            (bins, times, markers),
            select_frames=slice(None),
            aspect_range=(0.8, 1.2),
            frame_cluster=0,
        )
        assert shape == (256, 291)
        assert len(frame_markers) == 38
        assert frame_markers[0].tolist() == [4753, 78205]
        assert frame_markers[-1].tolist() == [2582185, 2635055]

        image = fbd.asimage(
            (bins, times, markers),
            (shape, frame_markers),
            integrate_frames=0,
            square_frame=True,
        )
        assert image.shape == (38, 4, 256, 256, 64)
        assert image.dtype == numpy.uint16
        assert image[0:, 0, 250, 10].sum(axis=(0, -1)) == 70
        counts = image.sum(axis=(0, 2, 3, 4), dtype=numpy.uint32)
        assert counts.tolist() == [693632, 0, 0, 0]

        # integrate frames
        image = fbd.asimage(
            (bins, times, markers),
            (shape, frame_markers),
            integrate_frames=1,
            square_frame=True,
        )
        assert image.shape == (1, 4, 256, 256, 64)
        assert image.dtype == numpy.uint16
        assert image[0, 0, 250, 10].sum() == 70

        if SHOW:
            imshow(image[:, 0].sum(-1, dtype=numpy.uint32), show=True)


def test_fbd_xx2x_b4w16c4():
    """Test read XX2X b4w16c4 FBD file."""
    # ISS VistaVision 32-bit with external fbs.xml settings
    # https://github.com/cgohlke/lfdfiles/issues/1
    fname = DATA / 'b4w16c4/E5+17+32M-20MHz-cell1$XX2X.fbd'
    if not os.path.exists(fname):
        pytest.skip(f'{fname!r} not found')

    with FbdFile(fname) as fbd:
        assert str(fbd).startswith(
            "<FbdFile 'E5+17+32M-20MHz-cell1$XX2X.fbd'>"
        )
        assert fbd.filename == os.path.basename(fname)
        assert fbd.dirname == os.path.dirname(fname)
        assert fbd.name == fbd.filename
        assert fbd.filehandle
        assert fbd.header is None
        assert fbd.fbf['channels'] == 4
        assert fbd.fbs['ScanParams']['Channels'] == 2
        assert fbd.decoder == '_b4w16c4t10'
        assert fbd.code == 'XX2X'
        assert fbd.frame_size == 256
        assert fbd.windows == 16
        assert fbd.channels == 4
        assert fbd.harmonics == 1
        assert fbd.pdiv == 4
        assert fbd.pixel_dwell_time == 20.0
        assert fbd.laser_frequency == 20000000.0
        assert fbd.laser_factor == 1.0
        assert fbd.scanner_line_length == 313
        assert fbd.scanner_line_start == 37
        assert fbd.scanner_frame_start == 0
        assert fbd.scanner == 'General LSM Scanner'
        assert fbd.synthesizer == 'Unknown'
        assert fbd.is_32bit

        bins, times, markers = fbd.decode(num_threads=6)

        assert bins.shape == (4, 3907584)
        assert bins.dtype == numpy.int8
        assert bins[1, 250000:250002].tolist() == [53, 51]
        hist = [numpy.bincount(b[b >= 0]) for b in bins]
        assert numpy.argmax(hist[1]) == 50

        assert times.shape == (3907584,)
        assert times.dtype == numpy.uint64
        assert times[[0, 1, -1]].tolist() == [0, 1024, 681550848]

        assert markers.shape == (21,)
        assert markers.dtype == numpy.int64
        assert markers[[0, -1]].tolist() == [18027, 3728761]

        shape, frame_markers = fbd.frames(
            (bins, times, markers),
            select_frames=slice(None),
            aspect_range=(0.8, 1.2),
            frame_cluster=0,
        )
        assert shape == (257, 313)
        assert len(frame_markers) == 20
        assert frame_markers[0].tolist() == [18027, 203143]
        assert frame_markers[-1].tolist() == [3543838, 3728760]

        image = fbd.asimage(
            (bins, times, markers),
            (shape, frame_markers),
            integrate_frames=0,
            square_frame=True,
            num_threads=6,
        )
        assert image.shape == (20, 4, 256, 256, 64)
        assert image.dtype == numpy.uint16
        assert image[0, 1, 128, 128].sum() == 11
        counts = image.sum(axis=(0, 2, 3, 4), dtype=numpy.uint32)
        assert counts.tolist() == [0, 2180641, 0, 593033]

        if SHOW:
            imshow(image[:, 1].sum(-1, dtype=numpy.uint32), show=True)


def test_fbd_ei0t_b4w8c4():
    """Test read EI0T b4w8c4 FBD file."""
    # SimFCS 32-bit with header settings; override pixel_dwell_time
    fname = (
        DATA
        / 'PhasorPy/60xw850fov48p30_cell3_nucb_mitogr_actinor_40f000$ei0t.fbd'
    )
    if not os.path.exists(fname):
        pytest.skip(f'{fname!r} not found')

    with FbdFile(fname, pixel_dwell_time=20.0) as fbd:
        assert str(fbd).startswith(
            "<FbdFile '60xw850fov48p30_cell3_nucb_mitogr_actinor_40f000$ei0t"
        )
        assert fbd.filename == os.path.basename(fname)
        assert fbd.dirname == os.path.dirname(fname)
        assert fbd.name == fbd.filename
        assert fbd.filehandle
        assert fbd.header['pixel_dwell_time_index'] == 0
        assert fbd.header['laser_factor'] == 1.000281
        assert fbd.fbf['channels'] == 4
        assert fbd.fbs is None
        assert fbd.decoder == '_b4w8c4'
        assert fbd.code == 'ei0t'
        assert fbd.frame_size == 256
        assert fbd.windows == 8
        assert fbd.channels == 4
        assert fbd.harmonics == 1
        assert fbd.pdiv == 4
        assert fbd.pixel_dwell_time == 20.0  # not 31.875
        assert fbd.laser_frequency == 80000000.0
        assert fbd.laser_factor == 1.000281
        assert fbd.scanner_line_length == 404
        assert fbd.scanner_line_start == 74
        assert fbd.scanner_frame_start == 0
        assert fbd.scanner == 'IOTech scanner card'
        assert fbd.synthesizer == 'Spectra Physics MaiTai'
        assert fbd.is_32bit

        bins, times, markers = fbd.decode()

        assert bins.shape == (4, 4640772)
        assert bins.dtype == numpy.int8
        assert bins[0, 2500:2502].tolist() == [56, 53]
        hist = [numpy.bincount(b[b >= 0]) for b in bins]
        assert numpy.argmax(hist[0]) == 55

        assert times.shape == (4640772,)
        assert times.dtype == numpy.uint64
        assert times[[0, 1, -1]].tolist() == [0, 8192, 10725434038]

        assert markers.shape == (41,)
        assert markers.dtype == numpy.int64
        assert markers[[0, -1]].tolist() == [37521, 4637325]

        shape, frame_markers = fbd.frames(
            (bins, times, markers),
            select_frames=slice(None),
            aspect_range=(0.8, 1.2),
            frame_cluster=0,
        )
        assert shape == (256, 404)
        assert len(frame_markers) == 40
        assert frame_markers[0].tolist() == [37521, 207501]

        image = fbd.asimage(
            (bins, times, markers),
            (shape, frame_markers),
            integrate_frames=0,
            square_frame=False,
        )
        assert image.shape == (40, 4, 257, 404, 64)
        assert image.dtype == numpy.uint16
        assert image[10, 0, 130, 176].sum() == 9
        counts = image.sum(axis=(0, 2, 3, 4), dtype=numpy.uint32)
        assert counts.tolist() == [1473312, 960165, 870617, 0]

        if SHOW:
            imshow(
                image.sum(axis=(0, -1), dtype=numpy.uint32),
                photometric='minisblack',
                show=True,
            )


def test_fbd_bytesio():
    """Test read FBD from BytesIO."""
    fname = DATA / 'CeruleanVenusCell1$CFCO.fbd'
    if not os.path.exists(fname):
        pytest.skip(f'{fname!r} not found')

    with open(fname, 'rb') as fh:
        data = io.BytesIO(fh.read())

    with FbdFile(data, code='CFCO') as fbd:
        assert str(fbd).startswith("<FbdFile 'BytesIO'>")
        assert fbd.filename == ''
        assert fbd.dirname == ''
        assert fbd.name == ''
        # assert fbd.name == fbd.filename
        assert fbd.filehandle
        assert fbd.header is None
        assert fbd.fbf is None
        assert fbd.fbs is None
        assert fbd.decoder == '_b2w8c2'
        assert fbd.code == 'CFCO'
        assert fbd.frame_size == 256
        assert fbd.windows == 8
        assert fbd.channels == 2
        assert fbd.harmonics == 1
        assert fbd.pdiv == 1
        assert fbd.pixel_dwell_time == 20.0
        assert fbd.laser_frequency == 20000000.0
        assert fbd.laser_factor == 1.0
        assert fbd.scanner_line_length == 312
        assert fbd.scanner_line_start == 55
        assert fbd.scanner_frame_start == 0
        assert fbd.scanner == 'Olympus FV 1000, NI USB'
        assert fbd.synthesizer == 'Unknown'
        assert not fbd.is_32bit

        image = fbd.asimage(integrate_frames=1, square_frame=True)
        assert image.shape == (1, 2, 256, 256, 64)
        assert image.dtype == numpy.uint16
        assert image[0, 0, 128, 128].sum() == 347
        counts = image.sum(axis=(0, 2, 3, 4), dtype=numpy.uint32)
        assert counts.tolist() == [2973148, 2642822]

        if SHOW:
            imshow(image.sum(-1, dtype=numpy.uint32), show=True)


@pytest.mark.parametrize('bytesio', [False, True])
def test_read_fbf(bytesio):
    """Test read FBF file."""
    fname = DATA / 'flimbox.fbf'
    if not os.path.exists(fname):
        pytest.skip(f'{fname!r} not found')

    if bytesio:
        with open(fname, 'rb') as fh:
            data = io.BytesIO(fh.read())
        fbf = fbf_read(data, firmware=True)
    else:
        fbf = fbf_read(fname, firmware=True)

    assert fbf['extclk']
    assert fbf['channels'] == 2
    assert fbf['windows'] == 16
    assert fbf['clkout'] == 10000000
    assert fbf['synchout'] == 10000000
    assert fbf['decoder'] == '16w2'
    assert fbf['fifofeedback'] == 0
    assert fbf['secondharmonic'] == 0
    assert fbf['optimalclk'] == 10000000
    assert fbf['comment'].startswith('Version 1.1.0 added channel select, ')
    assert fbf['firmware'][:4] == b'(\xecXP'


@pytest.mark.parametrize('stringio', [False, True])
def test_read_fbs(stringio):
    """Test read FBS file."""
    fname = DATA / 'FBS/TESTFILE.fbs.xml'
    if not os.path.exists(fname):
        pytest.skip(f'{fname!r} not found')

    if stringio:
        with open(fname, encoding='utf-8') as fh:
            data = io.StringIO(fh.read())
        fbs = fbs_read(data)
    else:
        fbs = fbs_read(fname)

    assert fbs['Comments'].startswith(
        'File created by ISS Vista software (Version: 4.2.597.0)'
    )
    assert fbs['DateTimeStamp'] == '2025-02-05T18:32:36.7762228-06:00'
    assert fbs['FirmwareParams']['ChannelMapping'] == (0, 1)
    assert fbs['FirmwareParams']['DecoderName'] == '8w'
    assert fbs['FirmwareParams']['Windows'] == 8
    assert fbs['FirmwareParams']['Use2ndHarmonic'] is False
    assert fbs['ScanParams']['Channels'] == 2
    assert fbs['ScanParams']['ExcitationFrequency'] == 40023631
    assert fbs['ScanParams']['FrameRepeat'] == 10
    assert fbs['SystemSettings']['fromComments'].startswith(
        '[Excitation Laser]\n'
    )


def test_sflim_decode():
    """Test sflim_decode and sflim_decode_photons functions."""
    fname = DATA / '20210123488_100x_NSC_166_TMRM_4_zoom4000_L115.bin'
    if not os.path.exists(fname):
        pytest.skip(f'{fname!r} not found')

    data = numpy.fromfile(fname, dtype=numpy.uint32)
    frequency = 78e6
    frequency_factor = 0.9976
    dwelltime = 16e-6
    pixeltime = numpy.ceil(
        dwelltime * 256 / 255 * frequency_factor * frequency
    )
    sflim = numpy.zeros((32, 256, 256, 342), dtype=numpy.uint8)
    sflim_decode(data, sflim, pixeltime=pixeltime, maxframes=20, num_threads=6)
    argmax = numpy.unravel_index(numpy.argmax(sflim), sflim.shape)
    assert_array_equal(argmax, (24, 178, 132, 248))

    del sflim

    photons = numpy.zeros((2035488, 5), dtype=numpy.uint16)
    nphotons = sflim_decode_photons(
        data, photons, (256, 342), pixeltime=pixeltime, maxframes=20
    )
    assert nphotons == 2035488
    assert photons[12345].tolist() == [205, 3, 2, 181, 51]


def test_fbd_to_b64():
    """Test fbd_to_b64 function."""
    fname = DATA / 'PhasorPy/cumarinech1_780LAURDAN_000$CC0Z.fbd'
    if not os.path.exists(fname):
        pytest.skip(f'{fname!r} not found')

    fbd_to_b64(
        fname,
        '{filename}_c{channel:02}t{frame:04}.b64',
        integrate_frames=1,
        square_frame=True,
        pdiv=-1,
        laser_frequency=-1,
        laser_factor=0.99168,
        pixel_dwell_time=-1.0,
        frame_size=-1,
        scanner_line_length=-1,
        scanner_line_start=-1,
        scanner_frame_start=-1,
        cmap='turbo',
        verbose=True,
        show=SHOW,
    )
    if not SHOW:
        with SimfcsB64(
            DATA / 'PhasorPy/cumarinech1_780LAURDAN_000$CC0Z.fbd_c00t0000.b64'
        ) as b64:
            assert b64.asarray().sum(dtype=numpy.int32) == 4287217  # 4312585
        with SimfcsB64(
            DATA / 'PhasorPy/cumarinech1_780LAURDAN_000$CC0Z.fbd_c01t0000.b64'
        ) as b64:
            assert b64.asarray().sum() == 0


def test_xml2dict():
    """Test xml2dict function."""
    xml = """<?xml version="1.0" ?>
    <root attr="attribute">
        <int>-1</int>
        <ints>-1,2</ints>
        <float>-3.14</float>
        <floats>1.0, -2.0</floats>
        <bool>True</bool>
        <string>Lorem, Ipsum</string>
    </root>
    """

    d = xml2dict(xml)['root']
    assert d['attr'] == 'attribute'
    assert d['int'] == -1
    assert d['ints'] == (-1, 2)
    assert d['float'] == -3.14
    assert d['floats'] == (1.0, -2.0)
    assert d['bool'] is True
    assert d['string'] == 'Lorem, Ipsum'

    d = xml2dict(xml, prefix=('a_', 'b_'), sep='')['root']
    assert d['ints'] == '-1,2'
    assert d['floats'] == '1.0, -2.0'


@pytest.mark.skipif(lfdfiles is None, reason='lfdfiles not installed')
def test_lfdfiles_fbd():
    """Test lfdfiles.FlimboxFbd class."""
    from lfdfiles import FlimboxFbd

    fname = DATA / 'PhasorPy/cumarinech1_780LAURDAN_000$CC0Z.fbd'
    if not os.path.exists(fname):
        pytest.skip(f'{fname!r} not found')

    with FlimboxFbd(fname, laser_factor=0.99168) as fbd:
        str(fbd)
        assert fbd.decoder == '_b2w4c2'
        assert fbd.laser_factor == 0.99168
        assert fbd.laser_frequency == 40000000.0
        assert fbd.pixel_dwell_time == 25.21
        assert fbd.header['laser_factor'] == 0.9955791

        bins, times, markers = fbd.decode()
        assert bins.shape == (2, 8380418)
        assert times.shape == (8380418,)
        assert markers.shape == (26,)

        bins = fbd.asarray()
        assert bins.shape == (2, 8380418)
        assert bins.dtype == numpy.int8
        assert bins.sum(dtype=numpy.uint32) == 117398019

        image = fbd.asimage((bins, times, markers), None)
        assert image.shape == (1, 2, 256, 256, 64)
        assert image.dtype == numpy.uint16
        assert image.sum(dtype=numpy.uint64) == 4287217

        with pytest.raises(AttributeError):
            fbd.non_existant

        if SHOW:
            fbd.show(cmap='turbo')


@pytest.mark.skipif(lfdfiles is None, reason='lfdfiles not installed')
def test_lfdfiles_fbf():
    """Test lfdfiles.FlimboxFbf class."""
    from lfdfiles import FlimboxFbf

    fname = DATA / 'flimbox.fbf'
    if not os.path.exists(fname):
        pytest.skip(f'{fname!r} not found')

    with FlimboxFbf(fname, firmware=True) as fbf:
        assert fbf['extclk']
        assert fbf['channels'] == 2
        assert fbf['windows'] == 16
        assert fbf['clkout'] == 10000000
        assert fbf['synchout'] == 10000000
        assert fbf['decoder'] == '16w2'
        assert fbf['fifofeedback'] == 0
        assert fbf['secondharmonic'] == 0
        assert fbf['optimalclk'] == 10000000
        assert fbf['comment'].startswith('Version 1.1.0 added channel')
        assert fbf.firmware()[:4] == b'(\xecXP'


@pytest.mark.skipif(lfdfiles is None, reason='lfdfiles not installed')
def test_lfdfiles_fbs():
    """Test lfdfiles.FlimboxFbs class."""
    from lfdfiles import FlimboxFbs

    fname = DATA / 'FBS/TESTFILE.fbs.xml'
    if not os.path.exists(fname):
        pytest.skip(f'{fname!r} not found')

    with FlimboxFbs(fname) as fbs:
        str(fbs)
        assert fbs['Comments'].startswith(
            'File created by ISS Vista software (Version: 4.2.597.0)'
        )
        assert fbs['DateTimeStamp'] == '2025-02-05T18:32:36.7762228-06:00'
        assert fbs['FirmwareParams']['ChannelMapping'] == (0, 1)
        assert fbs['FirmwareParams']['DecoderName'] == '8w'
        assert fbs['FirmwareParams']['Windows'] == 8
        assert fbs['FirmwareParams']['Use2ndHarmonic'] is False
        assert fbs['ScanParams']['Channels'] == 2
        assert fbs['ScanParams']['ExcitationFrequency'] == 40023631
        assert fbs['ScanParams']['FrameRepeat'] == 10
        assert fbs['SystemSettings']['fromComments'].startswith(
            '[Excitation Laser]\n'
        )


@pytest.mark.skipif(phasorpy is None, reason='phasorpy not installed')
def test_phasorpy():
    """Test phasorpy.io.signal_from_fbd function."""
    from phasorpy.datasets import fetch
    from phasorpy.io import signal_from_fbd

    filename = fetch('Convallaria_$EI0S.fbd')
    signal = signal_from_fbd(filename, channel=None, keepdims=True)
    assert signal.values.sum(dtype=numpy.uint64) == 9310275
    assert signal.dtype == numpy.uint16
    assert signal.shape == (9, 2, 256, 256, 64)
    assert signal.dims == ('T', 'C', 'Y', 'X', 'H')
    assert_almost_equal(
        signal.coords['H'].data[[1, -1]], [0.0981748, 6.1850105]
    )
    assert_almost_equal(signal.attrs['frequency'], 40.0)

    attrs = signal.attrs
    assert attrs['frequency'] == 40.0
    assert attrs['harmonic'] == 2
    assert attrs['flimbox_firmware']['secondharmonic'] == 1
    assert attrs['flimbox_header'] is not None
    assert 'flimbox_settings' not in attrs

    signal = signal_from_fbd(filename, frame=-1, channel=0, keepdims=True)
    assert signal.values.sum(dtype=numpy.uint64) == 9310275
    assert signal.shape == (1, 1, 256, 256, 64)
    assert signal.dims == ('T', 'C', 'Y', 'X', 'H')

    signal = signal_from_fbd(filename, frame=-1, channel=1)
    assert signal.values.sum(dtype=numpy.uint64) == 0  # channel 1 is empty
    assert signal.shape == (256, 256, 64)
    assert signal.dims == ('Y', 'X', 'H')

    signal = signal_from_fbd(filename, frame=1, channel=0, keepdims=False)
    assert signal.values.sum(dtype=numpy.uint64) == 1033137
    assert signal.shape == (256, 256, 64)
    assert signal.dims == ('Y', 'X', 'H')

    signal = signal_from_fbd(filename, frame=1, channel=0, keepdims=True)
    assert signal.values.sum(dtype=numpy.uint64) == 1033137
    assert signal.shape == (1, 1, 256, 256, 64)
    assert signal.dims == ('T', 'C', 'Y', 'X', 'H')

    with pytest.raises(IndexError):
        signal_from_fbd(filename, frame=9)

    with pytest.raises(IndexError):
        signal_from_fbd(filename, channel=2)

    # filename = fetch('simfcs.r64')
    # with pytest.raises(FbdFileError):
    #     signal_from_fbd(filename)


@pytest.mark.parametrize(
    'fname', glob.glob('**/*.fbf', root_dir=DATA, recursive=True)
)
def test_glob_fbf(fname):
    """Test read all FBF files."""
    if 'defective' in fname:
        pytest.xfail(reason='file is marked defective')
    fbf_read(DATA / fname)


@pytest.mark.parametrize(
    'fname', glob.glob('**/*.fbs.xml', root_dir=DATA, recursive=True)
)
def test_glob_fbs(fname):
    """Test read all FBS files."""
    if 'defective' in fname:
        pytest.xfail(reason='file is marked defective')
    fbs_read(DATA / fname)


@pytest.mark.parametrize(
    'fname', glob.glob('**/*.fbd', root_dir=DATA, recursive=True)
)
def test_glob_fbd(fname):
    """Test read all FBD files."""
    if 'defective' in fname:
        pytest.xfail(reason='file is marked defective')
    fname = DATA / fname
    with FbdFile(fname) as fbd:
        str(fbd)
        fbd.decode()
        fbd.asimage(None, None)
        fbd.plot(show=False)  # TODO: plots are not closed
    pyplot.close()


@pytest.mark.skipif(
    not hasattr(sys, '_is_gil_enabled'), reason='Python < 3.12'
)
def test_gil_enabled():
    """Test that GIL is disabled on thread-free Python."""
    assert sys._is_gil_enabled() != sysconfig.get_config_var('Py_GIL_DISABLED')


if __name__ == '__main__':
    import warnings

    # warnings.simplefilter('always')
    warnings.filterwarnings('ignore', category=ImportWarning)
    argv = sys.argv
    argv.append('--cov-report=html')
    argv.append('--cov=fbdfile')
    argv.append('--verbose')
    sys.exit(pytest.main(argv))

# mypy: allow-untyped-defs
# mypy: check-untyped-defs=False
