import 'dart:async';

import 'package:flutter_riverpod/flutter_riverpod.dart';

final playbackCoordinatorProvider = Provider((ref) => PlaybackCoordinator());

/// Native modal dialogs must not race cached review/import video playback.
/// Pausing preserves textures and positions; disposing a live Windows video
/// output here can itself deadlock the renderer.
class PlaybackCoordinator {
  PlaybackCoordinator({this.pauseTimeout = const Duration(seconds: 8)});

  final Duration pauseTimeout;
  final Map<Object, Future<void> Function()> _pausers = {};
  bool _dialogActive = false;

  bool get dialogActive => _dialogActive;

  void register(Object owner, Future<void> Function() pause) {
    _pausers[owner] = pause;
  }

  void unregister(Object owner) => _pausers.remove(owner);

  Future<T> withPausedPlayback<T>(Future<T> Function() showDialog) async {
    if (_dialogActive) throw StateError('An output dialog is already open');
    _dialogActive = true;
    try {
      await Future.wait([
        for (final pause in _pausers.values.toList()) pause(),
      ]).timeout(pauseTimeout);
      return await showDialog();
    } finally {
      // Do not automatically resume hidden players when a modal closes.
      _dialogActive = false;
    }
  }
}
