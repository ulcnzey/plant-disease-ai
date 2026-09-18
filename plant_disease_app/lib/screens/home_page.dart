import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../services/plant_classifier.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final ImagePicker _picker = ImagePicker();
  final PlantClassifier _classifier = PlantClassifier();

  File? _selectedImage;
  String? _result;
  double? _confidence;
  bool _loading = false;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      await _classifier.loadModel();
    } catch (e) {
      debugPrint('Model yükleme hatası: $e');
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    final XFile? image = await _picker.pickImage(
      source: source,
    );

    if (image == null) return;

    setState(() {
      _selectedImage = File(image.path);
      _result = null;
      _confidence = null;
      _loading = true;
    });

    try {
      final bytes = await _selectedImage!.readAsBytes();

      final prediction = await _classifier.predict(bytes);

      setState(() {
        _result = prediction['label'];
        _confidence = prediction['confidence'];
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _loading = false;
      });

      debugPrint('Tahmin hatası: $e');
    }
  }

  @override
  void dispose() {
    _classifier.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Plant Disease AI'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            Expanded(
              child: _selectedImage == null
                  ? const Center(
                      child: Text(
                        'Bir bitki fotoğrafı seçin',
                        style: TextStyle(fontSize: 20),
                      ),
                    )
                  : ClipRRect(
                      borderRadius: BorderRadius.circular(20),
                      child: Image.file(
                        _selectedImage!,
                        width: double.infinity,
                        fit: BoxFit.cover,
                      ),
                    ),
            ),

            const SizedBox(height: 20),

            if (_loading)
              const CircularProgressIndicator(),

            if (_result != null) ...[
              Text(
                _result!,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                'Güven: ${(_confidence! * 100).toStringAsFixed(2)}%',
                style: const TextStyle(fontSize: 17),
              ),
              const SizedBox(height: 20),
            ],

            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () => _pickImage(ImageSource.camera),
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Kamera'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () => _pickImage(ImageSource.gallery),
                    icon: const Icon(Icons.photo),
                    label: const Text('Galeri'),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}