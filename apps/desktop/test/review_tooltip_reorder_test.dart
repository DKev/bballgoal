import 'package:desktop/features/review/review_screen.dart';
import 'package:desktop/providers/project_state.dart';
import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';

ProjectState _state(int from) => ProjectState(
  job: const {'state': 'completed'},
  players: const [{'id': 'p1', 'name': 'A'}],
  candidates: [
    for (var i = from; i < 8; i++)
      {'id': 'c$i', 'event_time_ms': i * 3000 + 3000, 'default_start_ms': i * 3000, 'default_end_ms': i * 3000 + 3000},
  ],
);

class _N extends ProjectNotifier {
  @override
  ProjectState build() => _state(0);
  void drop() => state = _state(1);
  @override
  Future<void> startReview(String id) async {}
  @override
  Future<String?> loadCandidatePreview(String id, int t) async => null;
}

void main() {
  testWidgets('a row with an open tooltip can move when the list reorders', (tester) async {
    tester.view.physicalSize = const Size(1400, 900);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.reset);
    late _N notifier;
    await tester.pumpWidget(ProviderScope(
      overrides: [projectProvider.overrideWith(() => notifier = _N())],
      child: const MaterialApp(home: Scaffold(body: ReviewScreen())),
    ));
    await tester.pumpAndSettle();
    final mouse = await tester.createGesture(kind: PointerDeviceKind.mouse);
    await mouse.addPointer(location: tester.getCenter(find.byKey(const ValueKey('player-chip-c3'))));
    for (var i = 0; i < 20; i++) {
      await tester.pump(const Duration(milliseconds: 100));
    }
    expect(find.text('设置球员标签'), findsOneWidget);
    notifier.drop();
    for (var i = 0; i < 10; i++) {
      await tester.pump(const Duration(milliseconds: 50));
    }
    expect(tester.takeException(), isNull);
    expect(find.byKey(const ValueKey('player-chip-c3')), findsOneWidget);
  });

  testWidgets('fast scrolling under a resting mouse keeps hit testing working', (tester) async {
    tester.view.physicalSize = const Size(1400, 900);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.reset);
    await tester.pumpWidget(ProviderScope(
      overrides: [projectProvider.overrideWith(() => _LongN())],
      child: const MaterialApp(home: Scaffold(body: ReviewScreen())),
    ));
    await tester.pumpAndSettle();
    final chip = find.byKey(const ValueKey('player-chip-c2'));
    final pitch = tester.getCenter(find.byKey(const ValueKey('player-chip-c3'))).dy -
        tester.getCenter(chip).dy;
    final position = tester
        .state<ScrollableState>(find.ancestor(of: chip, matching: find.byType(Scrollable)).first)
        .position;
    final mouseAt = tester.getCenter(chip);
    position.jumpTo(60 * pitch);
    await tester.pumpAndSettle();
    final mouse = await tester.createGesture(kind: PointerDeviceKind.mouse);
    await mouse.addPointer(location: mouseAt);
    await tester.pump();
    // Scrolling up fast: a new row lands under the mouse and opens its
    // tooltip, then drops below the list in the very next frame.
    for (var jump = 2; jump >= 0; jump--) {
      position.jumpTo(jump * 20 * pitch);
      await tester.pump();
      expect(tester.takeException(), isNull);
    }
    await mouse.moveBy(const Offset(0, 5));
    await tester.pump();
    expect(tester.takeException(), isNull);
  });
}

class _LongN extends ProjectNotifier {
  @override
  ProjectState build() => ProjectState(
    job: const {'state': 'completed'},
    players: const [{'id': 'p1', 'name': 'A'}],
    candidates: [
      for (var i = 0; i < 90; i++)
        {'id': 'c$i', 'event_time_ms': i * 3000 + 3000, 'default_start_ms': i * 3000, 'default_end_ms': i * 3000 + 3000},
    ],
  );
  @override
  Future<void> startReview(String id) async {}
  @override
  Future<String?> loadCandidatePreview(String id, int t) async => null;
}
