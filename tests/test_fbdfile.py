# test_fbdfile.py

# Copyright (c) 2012-2026, Christoph Gohlke
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

:Version: 2026.8.30

"""

import contextlib
import glob
import io
import os
import pathlib
import sys
import sysconfig
import warnings

import numpy
import pytest
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
    imread,
    sflim_decode,
)
from fbdfile._fbdfile import sflim_decode_photons
from fbdfile.fbdfile import BinaryFile, xml2dict

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
    import fsspec
except ImportError:
    fsspec = None  # type: ignore[assignment]

try:
    import phasorpy
except ImportError:
    phasorpy = None  # type: ignore[assignment]


@pytest.mark.skipif(__doc__ is None, reason='__doc__ is None')
def test_version():
    """Assert fbdfile versions match docstrings."""
    ver = ':Version: ' + __version__
    assert __doc__ is not None
    assert fbdfile.__doc__ is not None
    assert ver in __doc__
    assert ver in fbdfile.__doc__


class TestBinaryFile:
    """Test BinaryFile with different file-like inputs."""

    def setup_method(self):
        self.filename = os.path.normpath(DATA / 'binary.bin')
        if not os.path.exists(self.filename):
            pytest.skip(f'{self.filename!r} not found')

    def validate(
        self,
        fh: BinaryFile,
        filepath: str | None = None,
        filename: str | None = None,
        dirname: str | None = None,
        name: str | None = None,
        *,
        closed: bool = True,
    ) -> None:
        """Assert BinaryFile attributes."""
        if filepath is None:
            filepath = self.filename
        if filename is None:
            filename = os.path.basename(self.filename)
        if dirname is None:
            dirname = os.path.dirname(self.filename)
        if name is None:
            name = fh.filename

        attrs = fh.attrs
        assert attrs['name'] == name
        assert attrs['filepath'] == filepath

        assert fh.filepath == filepath
        assert fh.filename == filename
        assert fh.dirname == dirname
        assert fh.name == name
        assert fh.closed is False
        assert fh.filesize == 256
        assert len(fh.filehandle.read()) == 256
        fh.filehandle.seek(10)
        assert fh.filehandle.tell() == 10
        assert fh.filehandle.read(1) == b'\n'
        fh.close()
        # underlying filehandle may still be be open if
        # BinaryFile was given an open filehandle
        assert fh._fh.closed is closed
        # BinaryFile always reports itself as closed after close() is called
        assert fh.closed

    def test_str(self):
        """Test BinaryFile with str path."""
        file = self.filename
        with BinaryFile(file) as fh:
            self.validate(fh, closed=True)

    def test_pathlib(self):
        """Test BinaryFile with pathlib.Path."""
        file = pathlib.Path(self.filename)
        with BinaryFile(file) as fh:
            self.validate(fh, closed=True)

    def test_open_file(self):
        """Test BinaryFile with open binary file."""
        with open(self.filename, 'rb') as fh, BinaryFile(fh) as bf:
            self.validate(bf, closed=False)

    def test_bytesio(self):
        """Test BinaryFile with BytesIO."""
        with open(self.filename, 'rb') as fh:
            file = io.BytesIO(fh.read())
        with BinaryFile(file) as fh:
            self.validate(
                fh,
                filepath='',
                filename='',
                dirname='',
                name='BytesIO',
                closed=False,
            )

    @pytest.mark.skipif(fsspec is None, reason='fsspec not installed')
    def test_fsspec_openfile(self):
        """Test BinaryFile with fsspec OpenFile."""
        file = fsspec.open(self.filename)
        with BinaryFile(file) as fh:
            self.validate(fh, closed=True)

    @pytest.mark.skipif(fsspec is None, reason='fsspec not installed')
    def test_fsspec_localfileopener(self):
        """Test BinaryFile with fsspec LocalFileOpener."""
        with fsspec.open(self.filename) as file, BinaryFile(file) as fh:
            self.validate(fh, closed=False)

    def test_text_file_fails(self):
        """Test BinaryFile with open text file fails."""
        with open(self.filename) as fh:  # noqa: SIM117
            with pytest.raises(TypeError):
                BinaryFile(fh)

    def test_file_extension_fails(self):
        """Test BinaryFile with wrong file extension fails."""
        ext = BinaryFile._ext
        BinaryFile._ext = {'.lif'}
        try:
            with pytest.raises(ValueError):
                BinaryFile(self.filename)
        finally:
            BinaryFile._ext = ext

    def test_file_not_seekable(self):
        """Test BinaryFile with non-seekable file fails."""

        class File:
            # mock file object without tell methods
            def seek(self):
                pass

        with pytest.raises(ValueError):
            BinaryFile(File())

    def test_openfile_not_seekable(self):
        """Test BinaryFile with non-seekable file fails."""

        class File:
            # mock fsspec OpenFile without seek/tell methods
            @staticmethod
            def open(*args, **kwargs):
                del args, kwargs
                return File()

        with pytest.raises(ValueError):
            BinaryFile(File())

    def test_invalid_object(self):
        """Test BinaryFile with invalid file object fails."""

        class File:
            # mock non-file object
            pass

        with pytest.raises(TypeError):
            BinaryFile(File())

    def test_invalid_mode(self):
        """Test BinaryFile with invalid mode fails."""
        with pytest.raises(ValueError):
            BinaryFile(self.filename, mode='ab')

    def test_memmap_disabled(self):
        """Test memmap=False (default) does not memory-map file."""
        with BinaryFile(self.filename) as fh:
            assert fh._mm is None
        with BinaryFile(self.filename, memmap=False) as fh:
            assert fh._mm is None

    def test_memmap_bytesio_ignored(self):
        """Test memmap=True is silently ignored for BytesIO."""
        with open(self.filename, 'rb') as f:
            data = f.read()
        with BinaryFile(io.BytesIO(data), memmap=True) as fh:
            assert fh._mm is None
            assert bytes(fh._read_at(0, 4)) == b'\x00\x01\x02\x03'

    def test_memmap_mmap_object(self):
        """Test BinaryFile accepts an existing mmap.mmap as file handle."""
        import mmap as mmap_module

        with open(self.filename, 'rb') as f:
            mm = mmap_module.mmap(
                f.fileno(), 0, access=mmap_module.ACCESS_READ
            )
        arr = None
        try:
            with BinaryFile(mm, memmap=True) as fh:
                assert fh._mm is mm
                assert fh._mv is not None
                assert fh.memmapped
                assert fh.filesize == 256
                assert bytes(fh._read_at(0, 4)) == b'\x00\x01\x02\x03'
                arr = fh._read_array(10, 3, numpy.uint8)
                assert numpy.array_equal(arr, [10, 11, 12])
        finally:
            del arr  # release mmap-backed array before closing mmap
            mm.close()

    @pytest.mark.parametrize('memmap', [False, True])
    def test_read_at(self, memmap):
        """Test _read_at returns bytes or memoryview at offset."""
        with BinaryFile(self.filename, memmap=memmap) as fh:
            data = fh._read_at(0, 4)
            assert isinstance(data, memoryview if memmap else bytes)
            assert bytes(data) == b'\x00\x01\x02\x03'
            data = fh._read_at(10, 3)
            assert bytes(data) == b'\x0a\x0b\x0c'

    @pytest.mark.parametrize('memmap', [False, True])
    def test_read_into(self, memmap):
        """Test _read_into reads bytes at offset into existing uint8 buffer."""
        with BinaryFile(self.filename, memmap=memmap) as fh:
            buf = numpy.empty(4, numpy.uint8)
            n = fh._read_into(0, buf)
            assert n == 4
            assert numpy.array_equal(buf, [0, 1, 2, 3])

            buf = numpy.empty(3, numpy.uint8)
            n = fh._read_into(10, buf)
            assert n == 3
            assert numpy.array_equal(buf, [10, 11, 12])

            # read past end of file returns only available bytes
            buf = numpy.empty(8, numpy.uint8)
            n = fh._read_into(252, buf)
            assert n == 4
            assert numpy.array_equal(buf[:4], [252, 253, 254, 255])

    @pytest.mark.parametrize('memmap', [False, True])
    def test_read_array(self, memmap):
        """Test _read_array returns array at offset; dtype, count=-1, copy."""
        dtype = numpy.dtype('uint8')
        with BinaryFile(self.filename, memmap=memmap) as fh:
            arr = fh._read_array(0, 4, dtype)
            assert arr.dtype == dtype
            assert numpy.array_equal(arr, [0, 1, 2, 3])
            arr = fh._read_array(10, 3, dtype)
            assert numpy.array_equal(arr, [10, 11, 12])
            if memmap:
                assert not arr.flags.writeable
                arr_copy = fh._read_array(0, 4, dtype, copy=True)
                assert arr_copy.flags.writeable
                arr_copy[0] = 99  # must not raise
            else:
                # multi-byte dtype: binary.bin bytes as uint16 LE pairs
                arr16 = fh._read_array(0, 4, numpy.dtype('<u2'))
                assert numpy.array_equal(
                    arr16, [0x0100, 0x0302, 0x0504, 0x0706]
                )
            # count=-1 reads to end of file
            arr_end = fh._read_array(252, -1, dtype)
            assert numpy.array_equal(arr_end, [252, 253, 254, 255])

    def test_read_array_writable_memmap(self, tmp_path):
        """Test  writable=False makes mmap RO; writable=True keeps it RW."""
        tmpfile = tmp_path / 'test.bin'
        with open(self.filename, 'rb') as src:
            tmpfile.write_bytes(src.read())
        dtype = numpy.dtype('uint8')
        with BinaryFile(tmpfile, mode='r+', memmap=True) as fh:
            arr_ro = fh._read_array(0, 4, dtype)
            assert not arr_ro.flags.writeable
            arr_rw = fh._read_array(0, 4, dtype, writable=True)
            assert arr_rw.flags.writeable

    def test_read_array_truncate(self, caplog):
        """Test truncate=False raises; True returns partial; None logs."""
        import logging

        with (
            BinaryFile(io.BytesIO(bytes(range(5)))) as fh,
            pytest.raises(ValueError, match='expected 10 items, got 5'),
        ):
            fh._read_array(0, 10, numpy.uint8, truncate=False)
        with BinaryFile(io.BytesIO(bytes(range(5)))) as fh:
            arr = fh._read_array(0, 10, numpy.uint8, truncate=True)
            assert len(arr) == 5
            assert numpy.array_equal(arr, [0, 1, 2, 3, 4])
        logger = BinaryFile.__module__.split('.')[0]
        with (
            BinaryFile(io.BytesIO(bytes(range(5)))) as fh,
            caplog.at_level(logging.ERROR, logger=logger),
        ):
            arr = fh._read_array(0, 10, numpy.uint8, truncate=None)
        assert len(arr) == 5
        assert 'expected 10 items, got 5' in caplog.text

    def test_read_array_mmap_alignment(self, tmp_path):
        """Test mmap path truncates partial element to full boundary."""
        # 7 bytes; requesting 4 uint16 (8 bytes); 3 complete elements fit
        tmpfile = tmp_path / 'align.bin'
        tmpfile.write_bytes(b'\x01\x00\x02\x00\x03\x00\xff')
        dtype = numpy.dtype('<u2')
        with BinaryFile(tmpfile, memmap=True) as fh:
            arr = fh._read_array(0, 4, dtype, truncate=True)
            assert len(arr) == 3
            assert numpy.array_equal(arr, [1, 2, 3])
            # truncate=False must raise even on mmap path
            with pytest.raises(ValueError, match='expected 4 items, got 3'):
                fh._read_array(0, 4, dtype, truncate=False)

    def test_read_array_invalid_offset(self):
        """Test _read_array raises ValueError for negative offset."""
        with BinaryFile(io.BytesIO(bytes(range(8)))) as fh:  # noqa: SIM117
            with pytest.raises(ValueError, match='offset'):
                fh._read_array(-1, 4, numpy.uint8)

    def test_read_array_invalid_count(self):
        """Test _read_array raises ValueError for count < -1."""
        with BinaryFile(io.BytesIO(bytes(range(8)))) as fh:  # noqa: SIM117
            with pytest.raises(ValueError, match='count'):
                fh._read_array(0, -2, numpy.uint8)

    def test_write_at(self):
        """Test _write_at writes bytes to file at given offset."""
        with open(self.filename, 'rb') as fh:
            data = fh.read()
        file = io.BytesIO(data)
        with BinaryFile(file, mode='r+') as fh:
            fh._write_at(5, b'\xaa\xbb\xcc')
            assert bytes(fh._read_at(4, 5)) == b'\x04\xaa\xbb\xcc\x08'
        # verify persistence after close
        assert file.getvalue()[5:8] == b'\xaa\xbb\xcc'

    def test_write_at_memmap(self, tmp_path):
        """Test _write_at writes via mmap; mode='r+' creates writable mmap."""
        tmpfile = tmp_path / 'test.bin'
        tmpfile.write_bytes(pathlib.Path(self.filename).read_bytes())
        with BinaryFile(tmpfile, mode='r+', memmap=True) as fh:
            assert fh._mm is not None
            assert fh.writable
            assert bytes(fh._read_at(0, 4)) == b'\x00\x01\x02\x03'
            fh._write_at(5, b'\xaa\xbb\xcc')
        with BinaryFile(tmpfile) as fh:
            assert bytes(fh._read_at(5, 3)) == b'\xaa\xbb\xcc'

    def test_memmapped(self):
        """Test memmapped property reflects memory-map state."""
        with BinaryFile(self.filename) as fh:
            assert not fh.memmapped
        with BinaryFile(self.filename, memmap=True) as fh:
            assert fh.memmapped

    def test_writable(self, tmp_path):
        """Test writable property reflects file open mode."""
        tmpfile = tmp_path / 'test.bin'
        tmpfile.write_bytes(b'\x00' * 4)
        with BinaryFile(tmpfile) as fh:
            assert not fh.writable
        with BinaryFile(tmpfile, mode='r+') as fh:
            assert fh.writable

    def test_name_setter(self):
        """Test name setter updates display name and attrs."""
        with BinaryFile(self.filename) as fh:
            fh.name = 'custom'
            assert fh.name == 'custom'
            assert fh.attrs['name'] == 'custom'

    def test_set_lock(self):
        """Test set_lock enables and disables RLock; no-op when mmap active."""
        import threading

        with BinaryFile(self.filename) as fh:
            assert isinstance(fh.lock, contextlib.nullcontext)
            fh.set_lock(True)
            assert isinstance(fh.lock, type(threading.RLock()))
            fh.set_lock(False)
            assert isinstance(fh.lock, contextlib.nullcontext)
        with BinaryFile(self.filename, memmap=True) as fh:
            lock_before = fh.lock
            fh.set_lock(True)
            assert fh.lock is lock_before

    def test_repr(self):
        """Test __repr__ includes class name and file name."""
        with BinaryFile(self.filename) as fh:
            r = repr(fh)
            assert r.startswith('<BinaryFile ')
            assert 'binary.bin' in r


class TestFbdFile:
    """Test FbdFile with different file-like inputs."""

    def setup_method(self):
        self.filename = os.path.normpath(DATA / 'CeruleanVenusCell1$CFCO.fbd')
        if not os.path.exists(self.filename):
            pytest.skip(f'{self.filename!r} not found')

    def validate(self, fbd: FbdFile) -> None:
        # assert FbdFile attributes
        assert fbd.header is None
        assert fbd.fbf is None
        assert fbd.fbs is None
        assert fbd.decoder == '_b2w8c2'
        assert fbd.code == 'CFCO'
        assert fbd.frame_size == 256
        assert fbd.windows == 8
        assert not fbd.is_32bit

        bins, times, markers = fbd.decode()
        assert bins.shape == (2, 10506240)
        assert bins[0, :2].tolist() == [26, -1]
        assert times.shape == (10506240,)
        assert times[:2].tolist() == [0, 115]
        assert markers.shape == (50,)
        assert markers[[0, -1]].tolist() == [185694, 10299672]

        shape, frame_markers = fbd.frames((bins, times, markers))
        assert shape == (256, 312)
        assert frame_markers[0].tolist() == [192126, 529435]

        image = fbd.asimage((bins, times, markers), (shape, frame_markers))
        assert image.shape == (1, 2, 256, 256, 64)
        assert image[0, 0, 128, 128].sum() == 337

        attrs = fbd.attrs
        assert attrs['name'] == fbd.name
        assert attrs['filepath'] == fbd.filepath
        assert attrs['channels'] == fbd.channels
        assert attrs['code'] == fbd.code
        assert attrs['decoder'] == fbd.decoder
        assert attrs['frame_size'] == fbd.frame_size
        assert attrs['harmonics'] == fbd.harmonics
        assert attrs['is_32bit'] == fbd.is_32bit
        assert attrs['laser_factor'] == fbd.laser_factor
        assert attrs['laser_frequency'] == fbd.laser_frequency
        assert attrs['pdiv'] == fbd.pdiv
        assert attrs['pixel_dwell_time'] == fbd.pixel_dwell_time
        assert attrs['pmax'] == fbd.pmax
        assert attrs['scanner'] == fbd.scanner
        assert attrs['scanner_frame_start'] == fbd.scanner_frame_start
        assert attrs['scanner_line_add'] == fbd.scanner_line_add
        assert attrs['scanner_line_length'] == fbd.scanner_line_length
        assert attrs['scanner_line_start'] == fbd.scanner_line_start
        assert attrs['synthesizer'] == fbd.synthesizer
        assert attrs['units_per_sample'] == fbd.units_per_sample
        assert attrs['windows'] == fbd.windows

    def test_str(self):
        """Test FbdFile with str path."""
        file = self.filename
        with FbdFile(file) as fbd:
            self.validate(fbd)

    def test_pathlib(self):
        """Test FbdFile with pathlib.Path."""
        file = pathlib.Path(self.filename)
        with FbdFile(file) as fbd:
            self.validate(fbd)

    def test_bytesio(self):
        """Test FbdFile with BytesIO."""
        with open(self.filename, 'rb') as fh:
            file = io.BytesIO(fh.read())
        with FbdFile(file, code='CFCO') as fbd:
            self.validate(fbd)

    @pytest.mark.skipif(fsspec is None, reason='fsspec not installed')
    def test_fsspec_openfile(self):
        """Test FbdFile with fsspec OpenFile."""
        file = fsspec.open(self.filename)
        with FbdFile(file) as fbd:
            self.validate(fbd)
        file.close()

    @pytest.mark.skipif(fsspec is None, reason='fsspec not installed')
    def test_fsspec_localfileopener(self):
        """Test FbdFile with fsspec LocalFileOpener."""
        with fsspec.open(self.filename) as file, FbdFile(file) as fbd:
            self.validate(fbd)


def test_fbd_error():
    """Test FbdFile errors."""
    filename = DATA / 'flimbox_firmware.fbf'
    if not os.path.exists(filename):
        pytest.skip(f'{filename!r} not found')

    with pytest.raises(FileNotFoundError):
        FbdFile('nonexistingfile.fbd')
    with pytest.raises(ValueError):
        FbdFile(filename)


@pytest.mark.parametrize('asxarray', [False, True])
def test_imread(asxarray):
    """Test imread function."""
    xarray = pytest.importorskip('xarray') if asxarray else None

    filename = DATA / 'flimbox_data$CBCO.fbd'
    if not os.path.exists(filename):
        pytest.skip(f'{filename!r} not found')

    image = imread(
        filename,
        memmap=True,
        pixel_dwell_time=0.937,
        word_count=500000,
        skip_words=1900000,
        select_frames=slice(None),
        aspect_range=(0.8, 1.2),
        frame_cluster=0,
        refine=False,
        integrate_frames=1,
        square_frame=True,
        asxarray=asxarray,
    )

    assert image.shape == (1, 2, 256, 256, 64)
    assert image.dtype == numpy.uint16
    counts = image.sum(axis=(0, 2, 3, 4), dtype=numpy.uint32)
    assert numpy.asarray(counts).tolist() == [72988, 0]

    if xarray is not None:
        assert isinstance(image, xarray.DataArray)
        assert image.coords['H'][-1] == pytest.approx(6.18501054)
        assert image.attrs['laser_factor'] == 1.0
        assert image.attrs['pixel_dwell_time'] == 0.937


@pytest.mark.parametrize('memmap', [False, True])
def test_fbd_cbco_b2w8c2(memmap):
    """Test read CBCO b2w8c2 FBD file."""
    # SimFCS 16-bit with code settings; does not correctly decode image
    filename = DATA / 'flimbox_data$CBCO.fbd'
    if not os.path.exists(filename):
        pytest.skip(f'{filename!r} not found')

    with FbdFile(filename, memmap=memmap) as fbd:
        assert str(fbd).startswith("<FbdFile 'flimbox_data$CBCO.fbd'>")
        assert fbd.filename == os.path.basename(filename)
        assert fbd.dirname == os.path.dirname(filename)
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


@pytest.mark.parametrize('memmap', [False, True])
def test_fbd_cbco_b2w8c2_override(memmap):
    """Test read CBCO b2w8c2 FBD file with overridden pixel_dwell_time."""
    # pixel_dwell_time=0.937 (= 4.0 * 0.23425) is the correct value for this
    # file; the code table maps 'O'/'B' to 4.0 µs which is wrong.
    # Supplying the known dwell time and refine=False suppresses all warnings.
    filename = DATA / 'flimbox_data$CBCO.fbd'
    if not os.path.exists(filename):
        pytest.skip(f'{filename!r} not found')

    with FbdFile(filename, memmap=memmap, pixel_dwell_time=0.937) as fbd:
        assert fbd.pixel_dwell_time == 0.937
        assert fbd.laser_factor == 1.0

        bins, times, markers = fbd.decode(
            word_count=500000, skip_words=1900000
        )

        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            shape, frame_markers = fbd.frames(
                (bins, times, markers),
                select_frames=slice(None),
                aspect_range=(0.8, 1.2),
                frame_cluster=0,
                refine=False,
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
        counts = image.sum(axis=(0, 2, 3, 4), dtype=numpy.uint32)
        assert counts.tolist() == [72988, 0]


@pytest.mark.parametrize('memmap', [False, True])
def test_fbd_ec0z_b2w4c2(memmap):
    """Test read EC0Z b2w4c2 FBD file."""
    # SimFCS 16-bit with header; pixel_dwell_time is computed from the
    # header frame_time (25.1175 us); the physical scanner does 257 lines
    # per frame. refine_settings makes a ~0.001% laser_factor adjustment
    # but leaves units_per_sample essentially unchanged (~1014.5).
    filename = (
        DATA / 'calibration_coumarin6_EtOH_25xNA1p4oil_780nm_'
        '2lzrpwr06282021POSTEXP$EC0Z.fbd'
    )
    if not os.path.exists(filename):
        pytest.skip(f'{filename!r} not found')

    with FbdFile(filename, memmap=memmap) as fbd:
        assert fbd.code == 'EC0Z'
        assert fbd.frame_size == 256
        assert fbd.windows == 4
        assert fbd.channels == 2
        assert fbd.harmonics == 2
        assert fbd.decoder == '_b2w4c2'
        assert (
            fbd.pixel_dwell_time == 25.1175
        )  # computed from header frame_time
        assert fbd.laser_factor == 1.00187
        assert fbd.scanner_line_length == 600
        assert fbd.scanner_line_start == 0
        assert fbd.scanner == 'Zeiss LSM710'
        assert not fbd.is_32bit

        bins, times, markers = fbd.decode()
        assert bins.shape == (2, 104843272)
        assert times[:2].tolist() == [0, 1024]
        assert markers.shape == (32,)
        assert markers[0] == 237992

        # default refine=True makes a ~0.001% laser_factor adjustment (noise)
        # below the 1/frame_size threshold, so no warning should be raised
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            shape, fm = fbd.frames((bins, times, markers))
        # 257 physical lines (one extra overhead line), frame_size=256
        assert shape == (257, 600)
        assert len(fm) == 15
        # units_per_sample stays ~1014.5; change is ~0.001%, not a correction
        assert abs(fbd.laser_factor - 1.00187) < 1e-4

        image = fbd.asimage((bins, times, markers), (shape, fm))
        assert image.shape == (1, 2, 256, 256, 64)
        counts = image.sum(axis=(0, 2, 3, 4), dtype=numpy.uint32)
        assert counts.tolist() == [55279245, 46940367]

    # refine=False: no warnings, same shape and frame count
    with FbdFile(filename) as fbd:
        bins, times, markers = fbd.decode()
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            shape, fm = fbd.frames((bins, times, markers), refine=False)
        assert shape == (257, 600)
        assert len(fm) == 15
        assert fbd.laser_factor == 1.00187  # unchanged


@pytest.mark.parametrize('memmap', [False, True])
def test_fbd_cfco_b2w8c2(memmap):
    """Test read CFCO b2w8c2 FBD file."""
    # SimFCS 16-bit with correct code settings
    filename = DATA / 'CeruleanVenusCell1$CFCO.fbd'
    if not os.path.exists(filename):
        pytest.skip(f'{filename!r} not found')

    with FbdFile(filename, memmap=memmap) as fbd:
        assert str(fbd).startswith("<FbdFile 'CeruleanVenusCell1$CFCO.fbd'>")
        assert fbd.filename == os.path.basename(filename)
        assert fbd.dirname == os.path.dirname(filename)
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
        assert image[0, 0, 128, 128].sum() == 11
        counts = image.sum(axis=(0, 2, 3, 4), dtype=numpy.uint32)
        assert counts.tolist() == [2973294, 2643017]

        if SHOW:
            imshow(image.sum(-1, dtype=numpy.uint32), show=True)


def test_fbd_xx2x_unknown_code():
    """Test that XX2X without fbs.xml raises FbdFileError for unknown code."""
    # Without an accompanying .fbs.xml, code[0]='X' is not in _frame_size
    # and should raise FbdFileError instead of a bare KeyError.
    filename = DATA / 'IFLItest/303 Cu6 vs FL$XX2X.fbd'
    if not os.path.exists(filename):
        pytest.skip(f'{filename!r} not found')

    import io

    with open(filename, 'rb') as f:
        data = io.BytesIO(f.read())
    data.name = filename.name

    # Open from BytesIO so no .fbs.xml is discovered on disk
    with pytest.raises(FbdFileError, match="unknown frame size code 'X'"):
        FbdFile(data)


@pytest.mark.parametrize('memmap', [False, True])
def test_fbd_xx2x_b4w16c4t10(memmap):
    """Test read XX2X b4w16c4t10 FBD file."""
    # ISS VistaVision 32-bit with external fbs.xml settings
    filename = DATA / 'IFLItest/303 Cu6 vs FL$XX2X.fbd'
    if not os.path.exists(filename):
        pytest.skip(f'{filename!r} not found')

    with FbdFile(filename, memmap=memmap) as fbd:
        assert str(fbd).startswith("<FbdFile '303 Cu6 vs FL$XX2X.fbd'>")
        assert fbd.filename == os.path.basename(filename)
        assert fbd.dirname == os.path.dirname(filename)
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
        assert image[0:, 0, 250, 10].sum(axis=(0, -1)) == 72
        counts = image.sum(axis=(0, 2, 3, 4), dtype=numpy.uint32)
        assert counts.tolist() == [693794, 0, 0, 0]

        # integrate frames
        image = fbd.asimage(
            (bins, times, markers),
            (shape, frame_markers),
            integrate_frames=1,
            square_frame=True,
        )
        assert image.shape == (1, 4, 256, 256, 64)
        assert image.dtype == numpy.uint16
        assert image[0, 0, 250, 10].sum() == 72

        if SHOW:
            imshow(image[:, 0].sum(-1, dtype=numpy.uint32), show=True)


@pytest.mark.parametrize('memmap', [False, True])
def test_fbd_xx2x_b4w16c4(memmap):
    """Test read XX2X b4w16c4 FBD file."""
    # ISS VistaVision 32-bit with external fbs.xml settings
    # https://github.com/cgohlke/lfdfiles/issues/1
    filename = DATA / 'b4w16c4/E5+17+32M-20MHz-cell1$XX2X.fbd'
    if not os.path.exists(filename):
        pytest.skip(f'{filename!r} not found')

    with FbdFile(filename, memmap=memmap) as fbd:
        assert str(fbd).startswith(
            "<FbdFile 'E5+17+32M-20MHz-cell1$XX2X.fbd'>"
        )
        assert fbd.filename == os.path.basename(filename)
        assert fbd.dirname == os.path.dirname(filename)
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
        assert shape == (257, 313)  # 257 physical scan lines (one overhead)
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
        assert counts.tolist() == [0, 2180677, 0, 593044]

        if SHOW:
            imshow(image[:, 1].sum(-1, dtype=numpy.uint32), show=True)


@pytest.mark.parametrize('memmap', [False, True])
def test_fbd_ei0t_b4w8c4(memmap):
    """Test read EI0T b4w8c4 FBD file."""
    # SimFCS 32-bit with header settings; override pixel_dwell_time
    filename = (
        DATA
        / 'PhasorPy/60xw850fov48p30_cell3_nucb_mitogr_actinor_40f000$ei0t.fbd'
    )
    if not os.path.exists(filename):
        pytest.skip(f'{filename!r} not found')

    with FbdFile(filename, memmap=memmap, pixel_dwell_time=20.0) as fbd:
        assert str(fbd).startswith(
            "<FbdFile '60xw850fov48p30_cell3_nucb_mitogr_actinor_40f000$ei0t"
        )
        assert fbd.filename == os.path.basename(filename)
        assert fbd.dirname == os.path.dirname(filename)
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
        assert fbd.pixel_dwell_time == 20.0  # not 32.0
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
        counts = image.sum(axis=(0, 2, 3, 4), dtype=numpy.uint32)
        assert counts.tolist() == [1473312, 960165, 870617, 0]

        if SHOW:
            imshow(
                image.sum(axis=(0, -1), dtype=numpy.uint32),
                photometric='minisblack',
                show=True,
            )


def test_fbd_ei0t_auto_refine():
    """Test refine_settings and refine for EI0T."""
    filename = (
        DATA / 'PhasorPy' / '60xw850fov48p30_cell3_nucb_mitogr_actinor'
        '_40f000$ei0t.fbd'
    )
    if not os.path.exists(filename):
        pytest.skip(f'{filename!r} not found')

    # header-derived dwell (32.0) already passes aspect check,
    # so refine is a no-op - verify frames still works
    with FbdFile(filename) as fbd:
        original_dwell = fbd.pixel_dwell_time
        records = fbd.decode()
        shape, fm = fbd.frames(records, refine=True)
        assert fbd.pixel_dwell_time == original_dwell
        assert shape == (256, 404)
        assert len(fm) == 40

    # user-supplied dwell is protected: refine_settings does not search
    # the table and emits a "no valid frames" warning instead
    with FbdFile(filename, pixel_dwell_time=100.0) as fbd:
        records = fbd.decode()
        assert not fbd._pixel_dwell_time_from_table
        with pytest.warns(UserWarning, match='does not produce valid frames'):
            fbd.refine_settings(records)
        assert fbd.pixel_dwell_time == 100.0  # unchanged

    # user-supplied laser_factor is protected: refine_settings skips update
    with FbdFile(filename, laser_factor=1.000281) as fbd:
        assert fbd._laser_factor_provided
        records = fbd.decode()
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            result = fbd.refine_settings(records)
        assert fbd.laser_factor == 1.000281  # unchanged because user-supplied
        assert result is False

    # table-derived dwell can be corrected: simulate by setting the flag
    with FbdFile(filename, pixel_dwell_time=100.0) as fbd:
        fbd._pixel_dwell_time_from_table = True
        records = fbd.decode()
        original_factor = fbd.laser_factor
        with pytest.warns(UserWarning, match='pixel_dwell_time changed'):
            fbd.refine_settings(records)
        assert fbd.pixel_dwell_time == 32.0
        assert fbd.laser_factor != original_factor

    # table-derived dwell corrected via frames() refine
    with FbdFile(filename, pixel_dwell_time=100.0) as fbd:
        fbd._pixel_dwell_time_from_table = True
        records = fbd.decode()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            shape, fm = fbd.frames(records, refine=True)
        assert fbd.pixel_dwell_time == 32.0
        assert shape == (256, 404)
        assert len(fm) == 40

    # table-derived dwell corrected via asimage() refine
    with FbdFile(filename, pixel_dwell_time=100.0) as fbd:
        fbd._pixel_dwell_time_from_table = True
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            image = fbd.asimage(
                refine=True,
                integrate_frames=0,
                square_frame=False,
            )
        assert fbd.pixel_dwell_time == 32.0
        assert image.shape == (40, 4, 257, 404, 64)
        counts = image.sum(axis=(0, 2, 3, 4), dtype=numpy.uint32)
        assert counts.tolist() == [1473312, 960165, 870617, 0]

    # refine=False skips refinement: bad dwell is not corrected
    with FbdFile(filename, pixel_dwell_time=100.0) as fbd:
        fbd._pixel_dwell_time_from_table = True
        records = fbd.decode()
        with pytest.warns(UserWarning, match='no frames detected'):
            fbd.frames(records, refine=False)
        assert fbd.pixel_dwell_time == 100.0  # unchanged

    # refine=None corrects bad dwell only when no frames detected
    with FbdFile(filename, pixel_dwell_time=100.0) as fbd:
        fbd._pixel_dwell_time_from_table = True
        records = fbd.decode()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            shape, fm = fbd.frames(records, refine=None)
        assert fbd.pixel_dwell_time == 32.0
        assert shape == (256, 404)
        assert len(fm) == 40

    # refine=None is a no-op when frames already detected (good dwell)
    with FbdFile(filename) as fbd:
        original_dwell = fbd.pixel_dwell_time
        original_factor = fbd.laser_factor
        records = fbd.decode()
        shape, fm = fbd.frames(records, refine=None)
        assert fbd.pixel_dwell_time == original_dwell
        assert fbd.laser_factor == original_factor
        assert shape == (256, 404)

    # refine_settings return values
    with FbdFile(filename, pixel_dwell_time=100.0) as fbd:
        fbd._pixel_dwell_time_from_table = True
        records = fbd.decode()
        # changing settings returns True
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = fbd.refine_settings(records)
        assert result is True

    with FbdFile(filename, pixel_dwell_time=100.0) as fbd:
        records = fbd.decode()
        # no valid frames with non-table source: returns None
        with pytest.warns(UserWarning, match='does not produce valid frames'):
            result = fbd.refine_settings(records)
        assert result is None

    with FbdFile(filename) as fbd:
        records = fbd.decode()
        # converges to stable value; may oscillate at floating-point precision
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            prev_factor = None
            for _ in range(20):
                fbd.refine_settings(records)
                if (
                    prev_factor is not None
                    and abs(fbd.laser_factor - prev_factor) < 1e-12
                ):
                    break
                prev_factor = fbd.laser_factor
        assert abs(fbd.laser_factor - 1.000281) < 1e-4


def test_fbd_issue1_ei0t_b2w4c2():
    """Test scan reconstruction fix for issue #1 (scanner_line_length)."""
    filename = (
        DATA / 'issue1/02_740nm_10x_8db_382u_1000zoom_ch1_B1C_1000$EI0T.fbd'
    )
    if not os.path.exists(filename):
        pytest.skip(f'{filename!r} not found')

    with FbdFile(filename, scanner_line_start=41) as fbd:
        assert fbd.decoder == '_b2w4c2'
        assert fbd.frame_size == 256
        assert fbd.windows == 4
        assert fbd.channels == 2
        assert fbd.harmonics == 2
        assert fbd.pdiv == 2
        assert fbd.pmax == 128
        assert fbd.laser_frequency == 40000000.0
        assert fbd.laser_factor == 0.99989
        assert fbd.scanner_line_length == 404
        assert fbd.scanner_line_start == 41
        assert fbd.scanner_frame_start == 0
        assert fbd.scanner == 'IOTech scanner card'
        assert fbd.synthesizer == 'Spectra Physics MaiTai'
        assert fbd.is_32bit is False

        bins, times, markers = fbd.decode()
        assert bins.shape == (2, 6893576)
        assert bins.dtype == numpy.int8
        assert times.shape == (6893576,)
        assert markers.shape == (37,)

        image = fbd.asimage(
            (bins, times, markers), integrate_frames=1, square_frame=True
        )
        assert image.shape == (1, 2, 256, 256, 64)
        assert image.dtype == numpy.uint16
        counts = image.sum(axis=(0, 2, 3, 4), dtype=numpy.uint32)
        assert counts.tolist() == [2477779, 2]


def test_fbd_bytesio():
    """Test read FBD from BytesIO."""
    filename = DATA / 'CeruleanVenusCell1$CFCO.fbd'
    if not os.path.exists(filename):
        pytest.skip(f'{filename!r} not found')

    with open(filename, 'rb') as fh:
        data = io.BytesIO(fh.read())

    with FbdFile(data, code='CFCO') as fbd:
        assert str(fbd).startswith("<FbdFile 'BytesIO'>")
        assert fbd.filename == ''
        assert fbd.dirname == ''
        assert fbd.name == 'BytesIO'
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
        assert image[0, 0, 128, 128].sum() == 337
        counts = image.sum(axis=(0, 2, 3, 4), dtype=numpy.uint32)
        assert counts.tolist() == [2973294, 2643017]

        if SHOW:
            imshow(image.sum(-1, dtype=numpy.uint32), show=True)


def test_fbd_cc0z_b2w4c2():
    """Test CC0Z b2w4c2 FBD file with auto laser_factor refinement."""
    # Fix #1: units_per_sample captured after frames() may update laser_factor.
    # Fix #2: integrate_frames exceeding frame count raises ValueError.
    # Fix #3: square_frame=True warns when detected line_num < frame_size.
    filename = DATA / 'PhasorPy/cumarinech1_780LAURDAN_000$CC0Z.fbd'
    if not os.path.exists(filename):
        pytest.skip(f'{filename!r} not found')

    with FbdFile(filename) as fbd:
        # header has slightly wrong laser_factor; frames() should refine it
        assert fbd.header['laser_factor'] == pytest.approx(0.9955791)
        initial_factor = fbd.laser_factor

        records = fbd.decode()
        _bins, _times, markers = records
        assert markers.shape == (26,)

        with pytest.warns(UserWarning, match='laser_factor changed'):
            frames = fbd.frames(records)

        # Fix #1: after frames(), laser_factor has been auto-refined
        assert fbd.laser_factor != initial_factor
        assert fbd.laser_factor == pytest.approx(0.99167, rel=1e-3)

        # Fix #1: asimage uses units_per_sample captured after frames() so the
        # histogram must match the reference value for the corrected factor
        image = fbd.asimage(
            records, frames, integrate_frames=1, square_frame=True
        )
        assert image.shape == (1, 2, 256, 256, 64)
        assert image.sum(dtype=numpy.uint64) == 4293221

        # Fix #2: integrate_frames > len(frame_markers) raises ValueError
        _, frame_markers = frames
        assert len(frame_markers) == 25
        with pytest.raises(ValueError, match='integrate_frames='):
            fbd.asimage(
                records,
                frames,
                integrate_frames=len(frame_markers) + 1,
            )

        # Fix #3: square_frame=True warns when detected line_num < frame_size;
        # use empty records and zero-frame markers so fbd_histogram is a no-op
        empty_records = fbd.decode(word_count=0)
        partial_frames = (
            (100, fbd.scanner_line_length),
            numpy.empty((0, 2), dtype=numpy.intp),
        )
        with pytest.warns(UserWarning, match='detected 100 lines'):
            fbd.asimage(
                empty_records,
                partial_frames,
                integrate_frames=0,
                square_frame=True,
            )


@pytest.mark.parametrize('bytesio', [False, True])
def test_read_fbf(bytesio):
    """Test read FBF file."""
    filename = DATA / 'flimbox_firmware.fbf'
    if not os.path.exists(filename):
        pytest.skip(f'{filename!r} not found')

    if bytesio:
        with open(filename, 'rb') as fh:
            data = io.BytesIO(fh.read())
        fbf = fbf_read(data, firmware=True)
    else:
        fbf = fbf_read(filename, firmware=True)

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
    filename = DATA / 'FBS/TESTFILE.fbs.xml'
    if not os.path.exists(filename):
        pytest.skip(f'{filename!r} not found')

    if stringio:
        with open(filename, encoding='utf-8') as fh:
            data = io.StringIO(fh.read())
        fbs = fbs_read(data)
    else:
        fbs = fbs_read(filename)

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


def test_fbd_asxarray():
    """Test asxarray on XX2X FBD file with FBS.XML settings."""
    xarray = pytest.importorskip('xarray')

    filename = DATA / 'FBS/TESTFILE$XX2X.fbd'
    if not os.path.exists(filename):
        pytest.skip(f'{filename!r} not found')

    with FbdFile(filename) as fbd:
        assert fbd.fbs is not None
        assert fbd.fbs['ScanParams']['Channels'] == 2
        assert fbd.windows == 8
        assert fbd.channels == 2

        xarr = fbd.asxarray(refine=False)

    assert isinstance(xarr, xarray.DataArray)
    assert xarr.dims == ('T', 'C', 'Y', 'X', 'H')
    assert xarr.shape == (1, 2, 254, 256, 64)
    assert xarr.dtype == numpy.uint16
    assert list(xarr.coords) == ['Y', 'X', 'H']
    assert float(xarr.coords['Y'][0]) == pytest.approx(-150.0)
    assert float(xarr.coords['Y'][-1]) == pytest.approx(150.0)
    assert float(xarr.coords['X'][0]) == pytest.approx(-150.0)
    assert float(xarr.coords['X'][-1]) == pytest.approx(150.0)
    assert float(xarr.coords['H'][0]) == pytest.approx(0.0)
    assert float(xarr.coords['H'][-1]) == pytest.approx(6.185011, rel=1e-4)
    assert xarr.attrs['windows'] == 8
    assert xarr.attrs['channels'] == 2
    assert xarr.attrs['frame_size'] == 256
    assert xarr.attrs['pmax'] == 64
    assert xarr.attrs['pdiv'] == 1
    assert xarr.attrs['laser_frequency'] == 40023631.0
    assert xarr.attrs['laser_factor'] == pytest.approx(1.0)
    assert xarr.attrs['pixel_dwell_time'] == pytest.approx(100.0)
    assert xarr.attrs['scanner'] == 'ISS Scanning Mirror'


def test_sflim_decode():
    """Test sflim_decode and sflim_decode_photons functions."""
    filename = DATA / '20210123488_100x_NSC_166_TMRM_4_zoom4000_L115.bin'
    if not os.path.exists(filename):
        pytest.skip(f'{filename!r} not found')

    data = numpy.fromfile(filename, dtype=numpy.uint32)
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
    lfdfiles = pytest.importorskip('lfdfiles')

    filename = DATA / 'PhasorPy/cumarinech1_780LAURDAN_000$CC0Z.fbd'
    if not os.path.exists(filename):
        pytest.skip(f'{filename!r} not found')

    fbd_to_b64(
        filename,
        '{filename}_c{channel:02}t{frame:04}.b64',
        integrate_frames=1,
        square_frame=True,
        pdiv=-1,
        laser_frequency=-1,
        laser_factor=0.99167,
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
        with lfdfiles.SimfcsB64(
            DATA / 'PhasorPy/cumarinech1_780LAURDAN_000$CC0Z.fbd_c00t0000.b64'
        ) as b64:
            assert b64.asarray().sum(dtype=numpy.int32) == 4293260
        with lfdfiles.SimfcsB64(
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


@pytest.mark.skipif(phasorpy is None, reason='phasorpy not installed')
def test_phasorpy():
    """Test phasorpy.io.signal_from_fbd function."""
    from phasorpy.datasets import fetch
    from phasorpy.io import signal_from_fbd

    filename = fetch('Convallaria_$EI0S.fbd')
    signal = signal_from_fbd(filename, channel=None, keepdims=True)
    assert signal.values.sum(dtype=numpy.uint64) == 9295075
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
    assert signal.values.sum(dtype=numpy.uint64) == 9295075
    assert signal.shape == (1, 1, 256, 256, 64)
    assert signal.dims == ('T', 'C', 'Y', 'X', 'H')

    signal = signal_from_fbd(filename, frame=-1, channel=1)
    assert signal.values.sum(dtype=numpy.uint64) == 0  # channel 1 is empty
    assert signal.shape == (256, 256, 64)
    assert signal.dims == ('Y', 'X', 'H')

    signal = signal_from_fbd(filename, frame=1, channel=0, keepdims=False)
    assert signal.values.sum(dtype=numpy.uint64) == 1031723
    assert signal.shape == (256, 256, 64)
    assert signal.dims == ('Y', 'X', 'H')

    signal = signal_from_fbd(filename, frame=1, channel=0, keepdims=True)
    assert signal.values.sum(dtype=numpy.uint64) == 1031723
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
    'filename', glob.glob('**/*.fbf', root_dir=DATA, recursive=True)
)
def test_glob_fbf(filename):
    """Test read all FBF files."""
    if 'defective' in filename:
        pytest.xfail(reason='file is marked defective')
    fbf_read(DATA / filename)


@pytest.mark.parametrize(
    'filename', glob.glob('**/*.fbs.xml', root_dir=DATA, recursive=True)
)
def test_glob_fbs(filename):
    """Test read all FBS files."""
    if 'defective' in filename:
        pytest.xfail(reason='file is marked defective')
    fbs_read(DATA / filename)


@pytest.mark.parametrize(
    'filename', glob.glob('**/*.fbd', root_dir=DATA, recursive=True)
)
def test_glob_fbd(filename):
    """Test read all FBD files."""
    if 'defective' in filename:
        pytest.xfail(reason='file is marked defective')
    filename = DATA / filename
    try:
        fbd = FbdFile(filename, memmap=True)
    except FbdFileError as exc:
        if 'unknown frame size code' in str(exc):
            pytest.xfail(reason=str(exc))
        raise
    with fbd:
        str(fbd)
        fbd.decode()
        fbd.asimage(None, None)
        fbd.plot(show=False)  # TODO: plots are not closed
    pyplot.close()


@pytest.mark.skipif(
    not hasattr(sys, '_is_gil_enabled'), reason='Python < 3.13'
)
def test_gil_enabled():
    """Test that GIL state is consistent with build configuration."""
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
