import 'package:desktop/features/review/clip_range_timeline.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

class _Harness {
  int windowStart = 0;
  int windowEnd = 30000;
  int start = 10000;
  int end = 20000;
  int position = 0;
  final scrubs = <int>[];
  final ended = <ClipRangeHandle>[];
  final extended = <ClipRangeHandle>[];
}

Future<_Harness> _pump(WidgetTester tester) async {
  final harness = _Harness();
  tester.view.physicalSize = const Size(640, 200);
  tester.view.devicePixelRatio = 1;
  addTearDown(tester.view.reset);
  await tester.pumpWidget(
    MaterialApp(
      home: Scaffold(
        body: StatefulBuilder(
          builder: (context, setState) => Center(
            child: SizedBox(
              width: 620,
              child: ClipRangeTimeline(
                windowStartMs: harness.windowStart,
                windowEndMs: harness.windowEnd,
                startMs: harness.start,
                endMs: harness.end,
                positionMs: harness.position,
                eventMs: 15000,
                originalStartMs: 10000,
                originalEndMs: 20000,
                enabled: true,
                onRangeChanged: (handle, start, end) => setState(() {
                  harness.start = start;
                  harness.end = end;
                }),
                onRangeChangeEnd: harness.ended.add,
                onScrub: (ms) => setState(() {
                  harness.position = ms;
                  harness.scrubs.add(ms);
                }),
                onScrubEnd: (_) {},
                onExtendWindow: (handle) => setState(() {
                  harness.extended.add(handle);
                  if (handle == ClipRangeHandle.end) harness.windowEnd += 5000;
                }),
              ),
            ),
          ),
        ),
      ),
    ),
  );
  return harness;
}

double _xAt(WidgetTester tester, int ms, int windowEnd) {
  final rect = tester.getRect(find.byType(ClipRangeTimeline));
  const pad = ClipRangeTimeline.padding;
  return rect.left + pad + (rect.width - pad * 2) * ms / windowEnd;
}

void main() {
  testWidgets('tapping the track moves the playhead', (tester) async {
    final harness = await _pump(tester);
    final rect = tester.getRect(find.byType(ClipRangeTimeline));
    await tester.tapAt(Offset(_xAt(tester, 25000, 30000), rect.center.dy));
    await tester.pump();
    expect(harness.position, closeTo(25000, 100));
    expect((harness.start, harness.end), (10000, 20000));
  });

  testWidgets('dragging a handle moves only that end', (tester) async {
    final harness = await _pump(tester);
    final rect = tester.getRect(find.byType(ClipRangeTimeline));
    final gesture = await tester.startGesture(
      Offset(_xAt(tester, 20000, 30000), rect.center.dy),
    );
    await gesture.moveBy(const Offset(20, 0));
    await gesture.moveTo(Offset(_xAt(tester, 24000, 30000), rect.center.dy));
    await gesture.up();
    await tester.pump();
    expect(harness.start, 10000);
    expect(harness.end, closeTo(24000, 100));
    expect(harness.ended, [ClipRangeHandle.end]);
    expect(harness.position, 0);
  });

  testWidgets('end handle cannot cross the start handle', (tester) async {
    final harness = await _pump(tester);
    final rect = tester.getRect(find.byType(ClipRangeTimeline));
    final gesture = await tester.startGesture(
      Offset(_xAt(tester, 20000, 30000), rect.center.dy),
    );
    await gesture.moveBy(const Offset(-20, 0));
    await gesture.moveTo(Offset(rect.left + 2, rect.center.dy));
    await gesture.up();
    await tester.pump();
    expect(harness.end - harness.start, 1000);
  });

  testWidgets('holding a handle at the edge keeps widening the axis', (
    tester,
  ) async {
    final harness = await _pump(tester);
    final rect = tester.getRect(find.byType(ClipRangeTimeline));
    final gesture = await tester.startGesture(
      Offset(_xAt(tester, 20000, 30000), rect.center.dy),
    );
    await gesture.moveBy(const Offset(20, 0));
    await gesture.moveTo(Offset(rect.right - 2, rect.center.dy));
    for (var i = 0; i < 3; i++) {
      await tester.pump(const Duration(milliseconds: 250));
    }
    await gesture.up();
    await tester.pump();
    expect(harness.extended.length, greaterThanOrEqualTo(2));
    expect(harness.windowEnd, greaterThan(30000));
    expect(harness.end, greaterThan(30000));
  });
}
