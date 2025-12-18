import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/connection_provider.dart';
import '../providers/run_provider.dart';
import '../services/storage_service.dart';
import '../widgets/connection_status.dart';
import '../widgets/model_picker.dart';
import '../widgets/folder_picker.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _formKey = GlobalKey<FormState>();
  String? _modelPath;
  String? _inputDir;
  String? _outputDir;
  double _confidence = 0.5;
  final TextEditingController _runNameController = TextEditingController();
  int _batchSize = 100;
  final StorageService _storageService = StorageService();

  @override
  void initState() {
    super.initState();
    _loadSettings();
  }

  Future<void> _loadSettings() async {
    final batchSize = await _storageService.getBatchSize();
    final savedModelPath = await _storageService.getModelPath();
    setState(() {
      _batchSize = batchSize;
      _modelPath = savedModelPath;
    });
  }

  @override
  void dispose() {
    _runNameController.dispose();
    super.dispose();
  }

  Future<void> _startProcessing() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }

    if (_modelPath == null || _inputDir == null || _outputDir == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Seleziona tutti i parametri richiesti'),
          backgroundColor: Colors.red,
        ),
      );
      return;
    }

    final runProvider = Provider.of<RunProvider>(context, listen: false);
    final connectionProvider =
        Provider.of<ConnectionProvider>(context, listen: false);

    if (!connectionProvider.isConnected) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Server non connesso. Avvia il server prima di procedere.'),
          backgroundColor: Colors.red,
        ),
      );
      return;
    }

    try {
      final runName = _runNameController.text.trim().isEmpty
          ? 'Elaborazione ${DateTime.now().toString().substring(0, 19)}'
          : _runNameController.text.trim();

      await runProvider.startRun(
        modelPath: _modelPath!,
        inputDir: _inputDir!,
        outputDir: _outputDir!,
        confidence: _confidence,
        runName: runName,
        batchSize: _batchSize,
      );

      if (mounted) {
        Navigator.pushNamed(context, '/progress');
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Errore: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Rilevatore Bright Spot'),
        actions: [
          IconButton(
            icon: const Icon(Icons.history),
            onPressed: () => Navigator.pushNamed(context, '/history'),
            tooltip: 'Storico',
          ),
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () => Navigator.pushNamed(context, '/settings'),
            tooltip: 'Impostazioni',
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const ConnectionStatus(),
              const SizedBox(height: 24),
              ModelPicker(
                selectedPath: _modelPath,
                onPathSelected: (path) async {
                  setState(() {
                    _modelPath = path;
                  });
                  await _storageService.saveModelPath(path);
                },
              ),
              const SizedBox(height: 24),
              FolderPicker(
                label: 'Cartella Input',
                selectedPath: _inputDir,
                onPathSelected: (path) {
                  setState(() {
                    _inputDir = path;
                  });
                },
              ),
              const SizedBox(height: 24),
              FolderPicker(
                label: 'Cartella Output',
                selectedPath: _outputDir,
                onPathSelected: (path) {
                  setState(() {
                    _outputDir = path;
                  });
                },
              ),
              const SizedBox(height: 24),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Confidenza Minima: ${(_confidence * 100).toStringAsFixed(0)}%',
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Slider(
                    value: _confidence,
                    min: 0.0,
                    max: 1.0,
                    divisions: 100,
                    label: '${(_confidence * 100).toStringAsFixed(0)}%',
                    onChanged: (value) {
                      setState(() {
                        _confidence = value;
                      });
                    },
                  ),
                ],
              ),
              const SizedBox(height: 24),
              TextFormField(
                controller: _runNameController,
                decoration: const InputDecoration(
                  labelText: 'Nome Elaborazione',
                  hintText: 'Es: Linea A - Batch 1',
                  border: OutlineInputBorder(),
                ),
                validator: (value) {
                  if (value == null || value.trim().isEmpty) {
                    return 'Inserisci un nome per l\'elaborazione';
                  }
                  return null;
                },
              ),
              const SizedBox(height: 32),
              Consumer<ConnectionProvider>(
                builder: (context, connectionProvider, child) {
                  final isEnabled = connectionProvider.isConnected &&
                      _modelPath != null &&
                      _inputDir != null &&
                      _outputDir != null;

                  return ElevatedButton.icon(
                    onPressed: isEnabled ? _startProcessing : null,
                    icon: const Icon(Icons.play_arrow),
                    label: const Text('Avvia Elaborazione'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      textStyle: const TextStyle(fontSize: 18),
                    ),
                  );
                },
              ),
            ],
          ),
        ),
      ),
    );
  }
}

