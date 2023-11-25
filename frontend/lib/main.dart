import "dart:io";

import "package:flutter/foundation.dart";
import "package:flutter/material.dart";
import "package:image_picker/image_picker.dart";

void main() {
  runApp(const FlowerClassificationApp());
}

class FlowerClassificationApp extends StatelessWidget {
  const FlowerClassificationApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: "Flower Classification",
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true,
      ),
      home: const FlowerClassificationHomePage(),
    );
  }
}

class FlowerClassificationHomePage extends StatefulWidget {
  const FlowerClassificationHomePage({super.key});

  @override
  State<FlowerClassificationHomePage> createState() => _FlowerClassificationHomePageState();
}

class _FlowerClassificationHomePageState extends State<FlowerClassificationHomePage> {
  XFile? _image;

  Future<void> _selectImage() async {
    final imagePicker = ImagePicker();
    final image = await imagePicker.pickImage(source: ImageSource.gallery);
    if (image == null) {
      return;
    }
    setState(() {
      _image = image;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text("Flower Classification Home Page"),
      ),
      body: Center(
        child: ImageView(image: _image),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () async => await _selectImage(),
        child: const Icon(Icons.add),
      ),
    );
  }
}

class ImageView extends StatelessWidget {
  const ImageView({super.key, this.image});

  final XFile? image;

  @override
  Widget build(BuildContext context) {
    if (image == null) {
      return const Text("No image selected.");
    }
    if (kIsWeb) {
      return Image.network(image!.path);
    } else {
      return Image.file(File(image!.path));
    }
  }
}
