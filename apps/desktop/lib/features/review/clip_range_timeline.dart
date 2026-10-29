import 'dart:async';

import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../theme/app_colors.dart';

enum ClipRangeHandle { start, end }

/// Range editor axis: the clip block, its two handles, the playhead, the shot
/// moment and the range the dialog opened with, on one scrubbable track.
class ClipRangeTimeline extends StatefulWidget {
  const ClipRangeTimeline({
    super.key,
    required this.windowStartMs,
    required this.windowEndMs,
    required this.startMs,
    required this.endMs,
    required this.positionMs,
    required this.enabled,
    required this.onRangeChanged,
    required this.onRangeChangeEnd,
    required this.onScrub,
    required this.onScrubEnd,
    required this.onExtendWindow,
    this.eventMs,
    this.originalStartMs,
    this.originalEndMs,
    this.minDurationMs = 1000,
    this.semanticLabel,
  });

  final int windowStartMs;
  final int windowEndMs;
  final int startMs;
  final int endMs;
  final int positionMs;
  final bool enabled;
  final int? eventMs;
  final int? originalStartMs;
  final int? originalEndMs;
  final int minDurationMs;
  final String? semanticLabel;
  final void Function(ClipRangeHandle handle, int startMs, int endMs)
  onRangeChanged;
  final ValueChanged<ClipRangeHandle> onRangeChangeEnd;
  final ValueChanged<int> onScrub;
  final ValueChanged<int> onScrubEnd;
  final ValueChanged<ClipRangeHandle> onExtendWindow;

  static const double height = 48;
  static const double padding = 10;
  static const double edgeZone = 14;
  static const double handleSlop = 14;

  @override
  State<ClipRangeTimeline> createState() => _ClipRangeTimelineState();
}

enum _DragTarget { start, end, playhead }

class _ClipRangeTimelineState extends State<ClipRangeTimeline> {
  _DragTarget? _drag;
  double _pointerX = 0;
  double _width = 0;
  Timer? _edgeTimer;
  MouseCursor _cursor = SystemMouseCursors.click;

  @override
  void dispose() {
    _edgeTimer?.cancel();
    super.dispose();
  }

  int get _span =>
      (widget.windowEndMs - widget.windowStartMs).clamp(1, 1 << 40).toInt();

  double _xAt(int ms) {
    final usable = _width - ClipRangeTimeline.padding * 2;
    final ratio = (ms - widget.windowStartMs) / _span;
    return ClipRangeTimeline.padding + ratio.clamp(0.0, 1.0) * usable;
  }

  int _msAt(double x) {
    final usable = (_width - ClipRangeTimeline.padding * 2).clamp(1.0, 1e9);
    final ratio = ((x - ClipRangeTimeline.padding) / usable).clamp(0.0, 1.0);
    return (widget.windowStartMs + ratio * _span).round();
  }

  _DragTarget _targetAt(double x) {
    final startDistance = (x - _xAt(widget.startMs)).abs();
    final endDistance = (x - _xAt(widget.endMs)).abs();
    if (startDistance <= ClipRangeTimeline.handleSlop ||
        endDistance <= ClipRangeTimeline.handleSlop) {
      return startDistance <= endDistance ? _DragTarget.start : _DragTarget.end;
    }
    return _DragTarget.playhead;
  }

  void _applyDrag() {
    final ms = _msAt(_pointerX);
    switch (_drag) {
      case _DragTarget.start:
        final start = ms
            .clamp(widget.windowStartMs, widget.endMs - widget.minDurationMs)
            .toInt();
        widget.onRangeChanged(ClipRangeHandle.start, start, widget.endMs);
      case _DragTarget.end:
        final end = ms
            .clamp(widget.startMs + widget.minDurationMs, widget.windowEndMs)
            .toInt();
        widget.onRangeChanged(ClipRangeHandle.end, widget.startMs, end);
      case _DragTarget.playhead:
        widget.onScrub(ms);
      case null:
        break;
    }
  }

  void _updateEdgeTimer() {
    final atStartEdge =
        _drag == _DragTarget.start &&
        _pointerX <= ClipRangeTimeline.padding + ClipRangeTimeline.edgeZone;
    final atEndEdge =
        _drag == _DragTarget.end &&
        _pointerX >=
            _width - ClipRangeTimeline.padding - ClipRangeTimeline.edgeZone;
    if (!atStartEdge && !atEndEdge) {
      _edgeTimer?.cancel();
      _edgeTimer = null;
      return;
    }
    _edgeTimer ??= Timer.periodic(const Duration(milliseconds: 220), (_) {
      if (!mounted || _drag == null) return;
      // Pin the handle to the already widened edge, then widen once more.
      _applyDrag();
      widget.onExtendWindow(
        _drag == _DragTarget.start
            ? ClipRangeHandle.start
            : ClipRangeHandle.end,
      );
    });
  }

  void _onDragStart(Offset position) {
    if (!widget.enabled) return;
    _pointerX = position.dx;
    _drag = _targetAt(position.dx);
    _applyDrag();
  }

  void _onDragUpdate(Offset position) {
    if (_drag == null) return;
    _pointerX = position.dx;
    _applyDrag();
    _updateEdgeTimer();
  }

  void _onDragEnd() {
    final drag = _drag;
    _drag = null;
    _edgeTimer?.cancel();
    _edgeTimer = null;
    switch (drag) {
      case _DragTarget.start:
        widget.onRangeChangeEnd(ClipRangeHandle.start);
      case _DragTarget.end:
        widget.onRangeChangeEnd(ClipRangeHandle.end);
      case _DragTarget.playhead:
        widget.onScrubEnd(_msAt(_pointerX));
      case null:
        break;
    }
  }

  void _onHover(Offset position) {
    final next = _targetAt(position.dx) == _DragTarget.playhead
        ? SystemMouseCursors.click
        : SystemMouseCursors.resizeColumn;
    if (next != _cursor) setState(() => _cursor = next);
  }

  @override
  Widget build(BuildContext context) {
    final c = AppColors.of(context);
    return Semantics(
      label: widget.semanticLabel,
      child: LayoutBuilder(
        builder: (context, constraints) {
          _width = constraints.maxWidth;
          return MouseRegion(
            cursor: widget.enabled ? _cursor : SystemMouseCursors.basic,
            onHover: (event) => _onHover(event.localPosition),
            child: GestureDetector(
              behavior: HitTestBehavior.opaque,
              // Pick the handle under the press, not where the drag slop ends.
              dragStartBehavior: DragStartBehavior.down,
              onTapUp: widget.enabled
                  ? (details) {
                      final ms = _msAt(details.localPosition.dx);
                      widget.onScrub(ms);
                      widget.onScrubEnd(ms);
                    }
                  : null,
              onHorizontalDragStart: widget.enabled
                  ? (details) => _onDragStart(details.localPosition)
                  : null,
              onHorizontalDragUpdate: widget.enabled
                  ? (details) => _onDragUpdate(details.localPosition)
                  : null,
              onHorizontalDragEnd: widget.enabled ? (_) => _onDragEnd() : null,
              onHorizontalDragCancel: widget.enabled ? _onDragEnd : null,
              child: CustomPaint(
                size: Size(constraints.maxWidth, ClipRangeTimeline.height),
                painter: _TimelinePainter(
                  startX: _xAt(widget.startMs),
                  endX: _xAt(widget.endMs),
                  playheadX: _xAt(widget.positionMs),
                  playheadVisible:
                      widget.positionMs >= widget.windowStartMs &&
                      widget.positionMs <= widget.windowEndMs,
                  playheadInside:
                      widget.positionMs >= widget.startMs &&
                      widget.positionMs <= widget.endMs,
                  eventX:
                      widget.eventMs == null ||
                          widget.eventMs! < widget.windowStartMs ||
                          widget.eventMs! > widget.windowEndMs
                      ? null
                      : _xAt(widget.eventMs!),
                  originalStartX: widget.originalStartMs == null
                      ? null
                      : _xAt(widget.originalStartMs!),
                  originalEndX: widget.originalEndMs == null
                      ? null
                      : _xAt(widget.originalEndMs!),
                  originalChanged:
                      widget.originalStartMs != widget.startMs ||
                      widget.originalEndMs != widget.endMs,
                  enabled: widget.enabled,
                  colors: c,
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}

class _TimelinePainter extends CustomPainter {
  _TimelinePainter({
    required this.startX,
    required this.endX,
    required this.playheadX,
    required this.playheadVisible,
    required this.playheadInside,
    required this.eventX,
    required this.originalStartX,
    required this.originalEndX,
    required this.originalChanged,
    required this.enabled,
    required this.colors,
  });

  final double startX;
  final double endX;
  final double playheadX;
  final bool playheadVisible;
  final bool playheadInside;
  final double? eventX;
  final double? originalStartX;
  final double? originalEndX;
  final bool originalChanged;
  final bool enabled;
  final AppColors colors;

  @override
  void paint(Canvas canvas, Size size) {
    const pad = ClipRangeTimeline.padding;
    final trackTop = 12.0;
    final trackBottom = size.height - 10;
    final accent = enabled ? colors.orange : colors.textTertiary;

    canvas.drawRRect(
      RRect.fromLTRBR(
        pad,
        trackTop + 8,
        size.width - pad,
        trackBottom - 8,
        const Radius.circular(3),
      ),
      Paint()..color = colors.surface3,
    );

    final block = RRect.fromLTRBR(
      startX,
      trackTop,
      endX,
      trackBottom,
      const Radius.circular(4),
    );
    canvas.drawRRect(block, Paint()..color = accent.withValues(alpha: 0.22));
    canvas.drawRRect(
      block,
      Paint()
        ..color = accent
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.5,
    );

    final originalStart = originalStartX;
    final originalEnd = originalEndX;
    if (originalChanged && originalStart != null && originalEnd != null) {
      final paint = Paint()
        ..color = colors.textSecondary
        ..strokeWidth = 2;
      final y = size.height - 3;
      for (var x = originalStart; x < originalEnd; x += 7) {
        canvas.drawLine(
          Offset(x, y),
          Offset((x + 4).clamp(x, originalEnd), y),
          paint,
        );
      }
    }

    final event = eventX;
    if (event != null) {
      final paint = Paint()..color = colors.warning;
      canvas.drawPath(
        Path()
          ..moveTo(event - 5, 2)
          ..lineTo(event + 5, 2)
          ..lineTo(event, 10)
          ..close(),
        paint,
      );
      canvas.drawLine(
        Offset(event, trackTop),
        Offset(event, trackBottom),
        Paint()
          ..color = colors.warning.withValues(alpha: 0.7)
          ..strokeWidth = 1,
      );
    }

    for (final x in [startX, endX]) {
      final handle = RRect.fromLTRBR(
        x - 4,
        trackTop - 2,
        x + 4,
        trackBottom + 2,
        const Radius.circular(3),
      );
      canvas.drawRRect(handle, Paint()..color = accent);
      final grip = Paint()
        ..color = Colors.white.withValues(alpha: 0.85)
        ..strokeWidth = 1;
      final mid = (trackTop + trackBottom) / 2;
      canvas.drawLine(Offset(x - 1.5, mid - 5), Offset(x - 1.5, mid + 5), grip);
      canvas.drawLine(Offset(x + 1.5, mid - 5), Offset(x + 1.5, mid + 5), grip);
    }

    if (playheadVisible) {
      final color = playheadInside ? colors.textPrimary : colors.warning;
      canvas.drawLine(
        Offset(playheadX, trackTop - 4),
        Offset(playheadX, trackBottom + 4),
        Paint()
          ..color = color
          ..strokeWidth = 2,
      );
      canvas.drawCircle(
        Offset(playheadX, trackTop - 5),
        4,
        Paint()..color = color,
      );
    }
  }

  @override
  bool shouldRepaint(_TimelinePainter old) =>
      old.startX != startX ||
      old.endX != endX ||
      old.playheadX != playheadX ||
      old.playheadVisible != playheadVisible ||
      old.playheadInside != playheadInside ||
      old.eventX != eventX ||
      old.originalStartX != originalStartX ||
      old.originalEndX != originalEndX ||
      old.originalChanged != originalChanged ||
      old.enabled != enabled ||
      old.colors != colors;
}

/// Whether a key event should be left to a focused text field.
bool clipRangeTextFieldFocused() =>
    FocusManager.instance.primaryFocus?.context
        ?.findAncestorWidgetOfExactType<EditableText>() !=
    null;

bool clipRangeCommandPressed() {
  final keyboard = HardwareKeyboard.instance;
  return keyboard.isControlPressed ||
      keyboard.isMetaPressed ||
      keyboard.isAltPressed;
}
