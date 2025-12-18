import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'dart:io';

class FolderPicker extends StatelessWidget {
  final String label;
  final String? selectedPath;
  final Function(String) onPathSelected;
  final bool isDirectory;

  const FolderPicker({
    super.key,
    required this.label,
    this.selectedPath,
    required this.onPathSelected,
    this.isDirectory = true,
  });

  Future<void> _pickPath(BuildContext context) async {
    try {
      if (isDirectory) {
        String? selectedDirectory = await FilePicker.platform.getDirectoryPath();
        if (selectedDirectory != null) {
          onPathSelected(selectedDirectory);
        }
      } else {
        FilePickerResult? result = await FilePicker.platform.pickFiles(
          type: FileType.custom,
          allowedExtensions: ['pt'],
        );
        if (result != null && result.files.single.path != null) {
          onPathSelected(result.files.single.path!);
        }
      }
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Errore nella selezione: $e')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: const TextStyle(
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
                  selectedPath ?? 'Nessuna selezione',
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
              onPressed: () => _pickPath(context),
              icon: const Icon(Icons.folder_open),
              label: const Text('Scegli'),
            ),
          ],
        ),
      ],
    );
  }
}

