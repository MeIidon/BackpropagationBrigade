import "dart:convert";
import "dart:io";
import "dart:isolate";

import "package:flutter/foundation.dart";
import "package:flutter/material.dart";
import 'package:http/http.dart' as http;
import "package:image/image.dart" show decodeImage;
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

typedef Pixel = List<int>;
typedef ImageData = List<List<Pixel>>;

class _FlowerClassificationHomePageState extends State<FlowerClassificationHomePage> {
  XFile? _image;
  Future<String>? _imageClassFuture;

  Future<String> _callBackend(String name, Uint8List bytes) async {
    final request = http.MultipartRequest(
      "POST",
      Uri.parse("http://localhost:5000/predict"),
    );
    request.files.add(
      http.MultipartFile.fromBytes(
        "image",
        bytes,
        filename: name,
      ),
    );
    final response = await request.send();
    if (response.statusCode != 200) {
      return Future.error("Failed to classify image.");
    }
    final result = await response.stream.bytesToString();
    return Future.value(result);
  }

  Future<void> _classifyImage(XFile image) async {
    final imageAsBytes = await image.readAsBytes();
    final imageClassFuture = _callBackend(image.name, imageAsBytes);
    setState(() {
      _imageClassFuture = imageClassFuture;
    });
  }

  Future<void> _selectImage() async {
    final imagePicker = ImagePicker();
    final image = await imagePicker.pickImage(source: ImageSource.gallery);
    if (image == null) {
      return;
    }
    setState(() {
      _image = image;
    });
    await _classifyImage(image);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text("Flower Classification Home Page"),
      ),
      body: ListView(
        children: [
          Center(
            child: FutureBuilder(
              future: _imageClassFuture,
              builder: (context, snapshot) {
                if (_imageClassFuture == null) {
                  return Container();
                }
                if (snapshot.connectionState != ConnectionState.done) {
                  return const CircularProgressIndicator();
                }
                if (snapshot.hasError) {
                  return Text(
                    "Error: ${snapshot.error}",
                    style: TextStyle(
                      color: Theme.of(context).colorScheme.error,
                    ),
                  );
                }
                return Text(
                  "Prediction: ${snapshot.data}",
                  style: Theme.of(context).textTheme.headlineSmall,
                );
              },
            ),
          ),
          ImageView(image: _image),
        ],
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
