import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';

class ModelPicker extends StatelessWidget {
  final String? selectedPath;
  final Function(String) onPathSelected;

  const ModelPicker({
    super.key,
    this.selectedPath,
    required this.onPathSelected,
  });

  Future<void> _pickModel(BuildContext context) async {
    try {
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['pt'],
      );
      if (result != null && result.files.single.path != null) {
        onPathSelected(result.files.single.path!);
      }
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Errore nella selezione del modello: $e')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Modello AI',
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            Expanded(
              child: Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  selectedPath ?? 'Nessun modello selezionato',
                  style: TextStyle(
                    color: selectedPath != null ? Colors.black87 : Colors.grey,
                  ),
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ),
            const SizedBox(width: 8),
            ElevatedButton.icon(
              onPressed: () => _pickModel(context),
              icon: const Icon(Icons.model_training),
              label: const Text('Scegli Modello'),
            ),
          ],
        ),
      ],
    );
  }
}

