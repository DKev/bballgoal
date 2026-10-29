import 'dart:async';
import 'package:flutter_test/flutter_test.dart';
import 'package:desktop/providers/playback_coordinator.dart';

void main() {
  test('waits for all previews before showing a native dialog', () async {
    final coordinator = PlaybackCoordinator();
    final paused = Completer<void>();
    final events = <String>[];
    coordinator.register(Object(), () async {
      events.add('pause');
      await paused.future;
    });
    final result = coordinator.withPausedPlayback(() async {
      events.add('dialog');
      return 'path';
    });
    expect(coordinator.dialogActive, isTrue);
    expect(events, ['pause']);
    paused.complete();
    expect(await result, 'path');
    expect(events, ['pause', 'dialog']);
    expect(coordinator.dialogActive, isFalse);
  });

  test('rejects overlapping dialogs and unlocks after cancellation', () async {
    final coordinator = PlaybackCoordinator();
    final selected = Completer<String?>();
    final first = coordinator.withPausedPlayback(() => selected.future);
    await expectLater(
      coordinator.withPausedPlayback(() async => 'duplicate'),
      throwsStateError,
    );
    selected.complete(null);
    expect(await first, isNull);
    expect(await coordinator.withPausedPlayback(() async => 'retry'), 'retry');
  });

  test(
    'pause timeout does not open a dialog or permanently lock export',
    () async {
      final coordinator = PlaybackCoordinator(
        pauseTimeout: const Duration(milliseconds: 10),
      );
      final owner = Object();
      final stalled = Completer<void>();
      coordinator.register(owner, () => stalled.future);
      var shown = false;
      await expectLater(
        coordinator.withPausedPlayback(() async {
          shown = true;
        }),
        throwsA(isA<TimeoutException>()),
      );
      expect(shown, isFalse);
      expect(coordinator.dialogActive, isFalse);
      coordinator.unregister(owner);
      stalled.complete();
      await coordinator.withPausedPlayback(() async {
        shown = true;
      });
      expect(shown, isTrue);
    },
  );

  test(
    'native dialog error releases the guard without resuming playback',
    () async {
      final coordinator = PlaybackCoordinator();
      var pauses = 0;
      coordinator.register(Object(), () async {
        pauses++;
      });
      await expectLater(
        coordinator.withPausedPlayback(() async {
          throw StateError('dialog failed');
        }),
        throwsStateError,
      );
      expect(pauses, 1);
      expect(coordinator.dialogActive, isFalse);
    },
  );
}
