// test/theme/app_colors_test.dart
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:desktop/theme/app_colors.dart';

void main() {
  test('dark brand colors match spec', () {
    expect(darkAppColors.amber, const Color(0xFFF0B35A));
    expect(darkAppColors.orange, const Color(0xFFFF6B2C));
    expect(darkAppColors.background, const Color(0xFF0B0E14));
    expect(darkAppColors.surface, const Color(0xFF121620));
  });
  test('semantic colors distinct', () {
    expect(darkAppColors.goal, const Color(0xFF3FB950));
    expect(darkAppColors.pending, const Color(0xFFE3B341));
    expect(darkAppColors.excluded, const Color(0xFF6B7280));
  });
  test('light orange darkened for contrast', () {
    expect(lightAppColors.orange, const Color(0xFFB93E0E));
  });
  test('buttons use the same bright orange in both themes', () {
    expect(lightAppColors.orangeFill, darkAppColors.orangeFill);
  });
  for (final (name, colors) in [('dark', darkAppColors), ('light', lightAppColors)]) {
    test('$name text tokens stay readable on surfaces', () {
      double contrast(Color a, Color b) {
        final la = a.computeLuminance();
        final lb = b.computeLuminance();
        return ((la > lb ? la : lb) + 0.05) / ((la > lb ? lb : la) + 0.05);
      }

      final text = {
        'textSecondary': colors.textSecondary,
        'textTertiary': colors.textTertiary,
        'orange': colors.orange,
        'goal': colors.goal,
        'warning': colors.warning,
        'error': colors.error,
        'data': colors.data,
        for (var i = 0; i < colors.players.length; i++) 'player$i': colors.players[i],
      };
      for (final background in [colors.surface, colors.surface2, colors.surface3]) {
        for (final entry in text.entries) {
          expect(
            contrast(entry.value, background),
            greaterThanOrEqualTo(4.5),
            reason: '${entry.key} on $background',
          );
        }
      }
      expect(contrast(colors.onOrange, colors.orangeFill), greaterThanOrEqualTo(4.5));
    });
  }
  testWidgets('of(context) returns theme extension', (tester) async {
    late AppColors captured;
    await tester.pumpWidget(
      MaterialApp(
        theme: ThemeData(extensions: [darkAppColors]),
        home: Builder(
          builder: (c) {
            captured = AppColors.of(c);
            return const SizedBox();
          },
        ),
      ),
    );
    expect(captured.amber, darkAppColors.amber);
  });
}
