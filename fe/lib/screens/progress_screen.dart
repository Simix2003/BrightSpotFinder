import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/run_provider.dart';
import '../widgets/stats_card.dart';

class ProgressScreen extends StatelessWidget {
  const ProgressScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Elaborazione in Corso'),
      ),
      body: Consumer<RunProvider>(
        builder: (context, runProvider, child) {
          final status = runProvider.currentStatus;

          if (status == null) {
            return const Center(
              child: CircularProgressIndicator(),
            );
          }

          return SingleChildScrollView(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      children: [
                        Text(
                          'Stato: ${_getStatusText(status.status)}',
                          style: const TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 16),
                        LinearProgressIndicator(
                          value: status.progress,
                          minHeight: 8,
                        ),
                        const SizedBox(height: 8),
                        Text(
                          '${(status.progress * 100).toStringAsFixed(1)}%',
                          style: const TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 16),
                        Text(
                          'Batch ${status.currentBatch} di ${status.totalBatches}',
                          style: const TextStyle(fontSize: 14),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 16),
                StatsCard(
                  stats: status.stats,
                  title: 'Statistiche Parziali',
                ),
                if (status.status == 'completed') ...[
                  const SizedBox(height: 16),
                  ElevatedButton.icon(
                    onPressed: () {
                      Navigator.pushReplacementNamed(context, '/results');
                    },
                    icon: const Icon(Icons.check_circle),
                    label: const Text('Vedi Risultati'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 16),
                    ),
                  ),
                ],
                if (status.status == 'failed') ...[
                  const SizedBox(height: 16),
                  Card(
                    color: Colors.red.shade50,
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Column(
                        children: [
                          const Icon(
                            Icons.error,
                            color: Colors.red,
                            size: 48,
                          ),
                          const SizedBox(height: 8),
                          const Text(
                            'Elaborazione Fallita',
                            style: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                              color: Colors.red,
                            ),
                          ),
                          if (runProvider.error != null) ...[
                            const SizedBox(height: 8),
                            Text(
                              runProvider.error!,
                              style: const TextStyle(color: Colors.red),
                            ),
                          ],
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: 16),
                  ElevatedButton.icon(
                    onPressed: () {
                      runProvider.reset();
                      Navigator.pop(context);
                    },
                    icon: const Icon(Icons.arrow_back),
                    label: const Text('Torna Indietro'),
                  ),
                ],
              ],
            ),
          );
        },
      ),
    );
  }

  String _getStatusText(String status) {
    switch (status) {
      case 'running':
        return 'In Elaborazione';
      case 'completed':
        return 'Completata';
      case 'failed':
        return 'Fallita';
      default:
        return status;
    }
  }
}

