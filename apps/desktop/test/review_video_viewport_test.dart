import 'package:desktop/features/review/review_video_viewport.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  for (final frame in [
    const Size(1920, 1080),
    const Size(1440, 1080),
    const Size(1080, 1920),
    Size.zero,
  ]) {
    for (final viewport in [const Size(954, 701), const Size(600, 300)]) {
      testWidgets('covers $viewport with $frame and aligned annotations', (
        tester,
      ) async {
        await tester.pumpWidget(
          Directionality(
            textDirection: TextDirection.ltr,
            child: Center(
              child: SizedBox.fromSize(
                size: viewport,
                child: ReviewVideoViewport(
                  frameSize: frame,
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      const ColoredBox(key: Key('video'), color: Colors.green),
                      const SizedBox.expand(key: Key('annotations')),
                      const Align(
                        alignment: Alignment(0.5, -0.5),
                        child: SizedBox(
                          key: Key('marker'),
                          width: 2,
                          height: 2,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        );

        final visible = tester.getRect(find.byType(ReviewVideoViewport));
        final video = tester.getRect(find.byKey(const Key('video')));
        final annotations = tester.getRect(
          find.byKey(const Key('annotations')),
        );
        expect(video.width, greaterThanOrEqualTo(visible.width));
        expect(video.height, greaterThanOrEqualTo(visible.height));
        expect(video.center, visible.center);
        expect(annotations, video);
        final aspect = frame.isEmpty ? 16 / 9 : frame.aspectRatio;
        expect(video.width / video.height, closeTo(aspect, 0.0001));
        final marker = tester.getCenter(find.byKey(const Key('marker')));
        expect(
          marker.dx,
          closeTo(video.left + (video.width - 2) * .75 + 1, .001),
        );
        expect(
          marker.dy,
          closeTo(video.top + (video.height - 2) * .25 + 1, .001),
        );
        expect(
          tester.widget<ClipRect>(find.byType(ClipRect)).clipBehavior,
          Clip.hardEdge,
        );
        expect(tester.takeException(), isNull);
      });
    }
  }
}
