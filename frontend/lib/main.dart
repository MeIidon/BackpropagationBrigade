import "package:flutter/material.dart";

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

class FlowerClassificationHomePage extends StatelessWidget {
  const FlowerClassificationHomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text("Flower Classification Home Page"),
      ),
      body: const Placeholder(),
    );
  }
}
