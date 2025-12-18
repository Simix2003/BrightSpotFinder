import 'package:flutter/material.dart';
import '../services/storage_service.dart';
import '../services/api_service.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _storageService = StorageService();
  final _batchSizeController = TextEditingController();
  final _serverPortController = TextEditingController();
  final _serverUrlController = TextEditingController();
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    _loadSettings();
  }

  Future<void> _loadSettings() async {
    setState(() {
      _isLoading = true;
    });

    final batchSize = await _storageService.getBatchSize();
    final serverPort = await _storageService.getServerPort();
    final serverUrl = await _storageService.getServerUrl();

    setState(() {
      _batchSizeController.text = batchSize.toString();
      _serverPortController.text = serverPort.toString();
      _serverUrlController.text = serverUrl;
      _isLoading = false;
    });
  }

  Future<void> _saveSettings() async {
    try {
      final batchSize = int.tryParse(_batchSizeController.text);
      final serverPort = int.tryParse(_serverPortController.text);

      if (batchSize == null || batchSize <= 0) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Dimensione batch non valida'),
            backgroundColor: Colors.red,
          ),
        );
        return;
      }

      if (serverPort == null || serverPort < 1 || serverPort > 65535) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Porta server non valida (1-65535)'),
            backgroundColor: Colors.red,
          ),
        );
        return;
      }

      await _storageService.saveBatchSize(batchSize);
      await _storageService.saveServerPort(serverPort);
      await _storageService.saveServerUrl(_serverUrlController.text);

      // Note: L'app deve essere riavviata per applicare le nuove impostazioni del server
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Impostazioni salvate. Riavvia l\'app per applicare le modifiche al server.'),
            backgroundColor: Colors.green,
            duration: Duration(seconds: 4),
          ),
        );
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
  void dispose() {
    _batchSizeController.dispose();
    _serverPortController.dispose();
    _serverUrlController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Impostazioni'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Configurazione Elaborazione',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 16),
                    TextField(
                      controller: _batchSizeController,
                      decoration: const InputDecoration(
                        labelText: 'Dimensione Batch',
                        hintText: '100',
                        border: OutlineInputBorder(),
                        helperText:
                            'Numero di immagini processate per batch',
                      ),
                      keyboardType: TextInputType.number,
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Configurazione Server',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 16),
                    TextField(
                      controller: _serverUrlController,
                      decoration: const InputDecoration(
                        labelText: 'URL Server',
                        hintText: 'http://localhost:5000',
                        border: OutlineInputBorder(),
                        helperText: 'URL completo del server Python',
                      ),
                    ),
                    const SizedBox(height: 16),
                    TextField(
                      controller: _serverPortController,
                      decoration: const InputDecoration(
                        labelText: 'Porta Server',
                        hintText: '5000',
                        border: OutlineInputBorder(),
                        helperText: 'Porta del server (1-65535)',
                      ),
                      keyboardType: TextInputType.number,
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),
            ElevatedButton.icon(
              onPressed: _saveSettings,
              icon: const Icon(Icons.save),
              label: const Text('Salva Impostazioni'),
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 16),
                textStyle: const TextStyle(fontSize: 18),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

