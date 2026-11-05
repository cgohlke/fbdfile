..
  This file is updated by setup.py. Do not edit.

Revisions
=========

2026.11.5

- Fix histogram including samples before scanner_frame_start in first pixel.
- Fix frames excluding last data word of each frame (breaking).
- Fix histogram dropping last bin if pdiv does not divide pmax.
- Fix parsing FBF headers with unnamed or fractional MHz values.
- Fix BinaryFile._read_into on partial reads.
- Fix sflim_decode return value and saturate counts.
- Fix FbdFile.decode silently ignoring unknown arguments (breaking).
- Fix decoder_settings not updating when decoder changes.
- Fall back to defaults for invalid laser frequency and factor in header.
- Validate indices in FBD header.
- Add FBDFILE_NUM_THREADS environment variable.
- Parallelize fbd_decode over data words instead of channels.

2026.10.16

- Fix frames fallback overwriting user-provided laser_factor.
- Fix frames fallback computing relative instead of absolute laser_factor.
- Fix FBS settings overriding user-provided scanner_frame_start.
- Fix reading FBD files with embedded header from streams without name.
- Fix decode returning invalid markers for empty data.
- Fix potential out-of-bounds access and division by zero in Cython code.
- Saturate histogram counts at 65535 and warn instead of wrapping.

2026.8.30

- Do not refine user-provided laser_factor (breaking; see PhasorPy issue 372).
- Add FbdFile.asxarray method (requires xarray).
- Add imread convenience function.

2026.8.8

- Update BinaryFile class.

2026.6.6

- Fix potential file handle leak in FbdFile init.
- Fix pixel_dwell_time when scanner_line_start > 0.
- Add options for memory-mapping and locked reading to BinaryFile.
- Add option to memory-map FBD files.
- Drop support for numpy 2.0 (SPEC0).
- Support Python 3.15.

2026.3.20

- Frames and asimage methods always refine laser_factor by default (breaking).
- Add tri-state refine option to frames and asimage methods.
- Add refine_settings method to refine pixel_dwell_time and laser_factor.
- Add more decoder settings (not tested).
- Fix decode to read all bytes from streams and fix skip/count guard.
- Fix cluster-to-frame-marker mapping in frames fallback path.
- Fix from_fbs header detection to use fbf_parse_header.
- Fix decoder_settings to not swallow errors from valid decoders.
- Use fbd_decode return value to trim markers array.
- Drop support for Python 3.11.

2026.2.6

- Fix code review issues.

2026.1.14

- Improve code quality.

2025.12.12

- Add attrs property to FbdFile.
- Improve code quality.

2025.11.8

- Allow to override FbdFile decoder, firmware, and settings.
- Always try to load settings from .fbs.xml file.
- Factor out BinaryFile base class.
- Derive FbdFileError from ValueError.
- Build ABI3 wheels.

2025.9.18

- Fix reading FBF and FBS files from streams.

2025.9.17

- Make frame_markers a numpy array.
- Add options to specify number of OpenMP threads.

2025.9.16

- Initial alpha release based on lfdfiles 2025.7.31.
