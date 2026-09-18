
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class PlantClassifier {
  late Interpreter _interpreter;
  late List<String> _labels;

  Future<void> loadModel() async {
    _interpreter = await Interpreter.fromAsset('assets/model.tflite');

    final labelsData = await rootBundle.loadString('assets/labels.txt');

    _labels = labelsData
        .split('\n')
        .map((label) => label.trim())
        .where((label) => label.isNotEmpty)
        .toList();
  }

  Future<Map<String, dynamic>> predict(Uint8List imageBytes) async {
    final image = img.decodeImage(imageBytes);

    if (image == null) {
      throw Exception('Görüntü okunamadı.');
    }

    final resizedImage = img.copyResize(
      image,
      width: 224,
      height: 224,
    );

    final input = List.generate(
      1,
      (_) => List.generate(
        224,
        (y) => List.generate(
          224,
          (x) {
            final pixel = resizedImage.getPixel(x, y);

            return [
              pixel.r.toDouble(),
              pixel.g.toDouble(),
              pixel.b.toDouble(),
            ];
          },
        ),
      ),
    );

    final output = List.generate(
      1,
      (_) => List.filled(_labels.length, 0.0),
    );

    _interpreter.run(input, output);

    final probabilities = output[0];

    int bestIndex = 0;

    for (int i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > probabilities[bestIndex]) {
        bestIndex = i;
      }
    }

    return {
      'label': _labels[bestIndex],
      'confidence': probabilities[bestIndex],
      'index': bestIndex,
    };
  }

  void dispose() {
    _interpreter.close();
  }
}