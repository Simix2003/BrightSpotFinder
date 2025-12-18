import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:intl/intl.dart';
import '../providers/history_provider.dart';
import '../services/api_service.dart';
import '../models/stats_model.dart';
import '../widgets/stats_card.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      Provider.of<HistoryProvider>(context, listen: false).loadHistory();
    });
  }

  void _showCombinedStats(BuildContext context, List<String> runIds) async {
    if (runIds.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Seleziona almeno una run')),
      );
      return;
    }

    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => const Center(child: CircularProgressIndicator()),
    );

    try {
      final apiService = ApiService();
      final combinedStats = await apiService.getCombinedStats(runIds);

      if (context.mounted) {
        Navigator.pop(context); // Close loading dialog
        _showCombinedStatsDialog(context, combinedStats);
      }
    } catch (e) {
      if (context.mounted) {
        Navigator.pop(context); // Close loading dialog
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Errore: $e')),
        );
      }
    }
  }

  void _showCombinedStatsDialog(
      BuildContext context, CombinedStatsResponse stats) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Statistiche Combinate'),
        content: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('Run selezionate: ${stats.totalRuns}'),
              const Divider(),
              Text('Immagini Totali: ${stats.totalImagesProcessed}'),
              Text('Bright Spot Trovati: ${stats.totalBrightSpotsFound}'),
              Text(
                  'Senza Bright Spot: ${stats.totalImagesWithoutBrightSpot}'),
              Text(
                  'Tasso Medio Successo: ${stats.averageSuccessRate.toStringAsFixed(1)}%'),
              Text(
                  'Tasso Medio Bright Spot: ${stats.averageBrightSpotRate.toStringAsFixed(1)}%'),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Chiudi'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Storico Elaborazioni'),
        actions: [
          Consumer<HistoryProvider>(
            builder: (context, historyProvider, child) {
              return IconButton(
                icon: const Icon(Icons.refresh),
                onPressed: () => historyProvider.loadHistory(),
                tooltip: 'Aggiorna',
              );
            },
          ),
        ],
      ),
      body: Consumer<HistoryProvider>(
        builder: (context, historyProvider, child) {
          if (historyProvider.isLoading) {
            return const Center(child: CircularProgressIndicator());
          }

          if (historyProvider.error != null) {
            return Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(Icons.error, size: 64, color: Colors.red),
                  const SizedBox(height: 16),
                  Text('Errore: ${historyProvider.error}'),
                  const SizedBox(height: 16),
                  ElevatedButton(
                    onPressed: () => historyProvider.loadHistory(),
                    child: const Text('Riprova'),
                  ),
                ],
              ),
            );
          }

          if (historyProvider.runs.isEmpty) {
            return const Center(
              child: Text('Nessuna elaborazione trovata'),
            );
          }

          return Column(
            children: [
              if (historyProvider.hasSelection)
                Container(
                  padding: const EdgeInsets.all(8),
                  color: Colors.blue.shade50,
                  child: Row(
                    children: [
                      Expanded(
                        child: Text(
                          '${historyProvider.selectedRunIds.length} run selezionate',
                          style: const TextStyle(fontWeight: FontWeight.bold),
                        ),
                      ),
                      TextButton(
                        onPressed: () => historyProvider.deselectAll(),
                        child: const Text('Deseleziona Tutto'),
                      ),
                      ElevatedButton.icon(
                        onPressed: () => _showCombinedStats(
                          context,
                          historyProvider.selectedRunIds.toList(),
                        ),
                        icon: const Icon(Icons.analytics),
                        label: const Text('Statistiche Combinate'),
                      ),
                    ],
                  ),
                ),
              Expanded(
                child: ListView.builder(
                  itemCount: historyProvider.runs.length,
                  itemBuilder: (context, index) {
                    final run = historyProvider.runs[index];
                    final isSelected =
                        historyProvider.isSelected(run.runId);
                    final dateFormat = DateFormat('dd/MM/yyyy HH:mm');

                    return Card(
                      margin: const EdgeInsets.symmetric(
                          horizontal: 16, vertical: 8),
                      child: CheckboxListTile(
                        value: isSelected,
                        onChanged: (value) {
                          historyProvider.toggleRunSelection(run.runId);
                        },
                        title: Text(run.runName),
                        subtitle: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(dateFormat.format(run.timestamp)),
                            Text('Stato: ${_getStatusText(run.status)}'),
                            Text(
                                'Immagini: ${run.stats.totalImages} | Bright Spot: ${run.stats.imagesWithBrightSpot}'),
                          ],
                        ),
                        secondary: IconButton(
                          icon: const Icon(Icons.info),
                          onPressed: () {
                            _showRunDetails(context, run);
                          },
                        ),
                      ),
                    );
                  },
                ),
              ),
              Container(
                padding: const EdgeInsets.all(8),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    TextButton.icon(
                      onPressed: () => historyProvider.selectAll(),
                      icon: const Icon(Icons.select_all),
                      label: const Text('Seleziona Tutto'),
                    ),
                    TextButton.icon(
                      onPressed: () => historyProvider.deselectAll(),
                      icon: const Icon(Icons.deselect),
                      label: const Text('Deseleziona Tutto'),
                    ),
                  ],
                ),
              ),
            ],
          );
        },
      ),
    );
  }

  void _showRunDetails(BuildContext context, run) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(run.runName),
        content: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              StatsCard(stats: run.stats),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Chiudi'),
          ),
        ],
      ),
    );
  }

  String _getStatusText(String status) {
    switch (status) {
      case 'completed':
        return 'Completata';
      case 'running':
        return 'In Elaborazione';
      case 'failed':
        return 'Fallita';
      default:
        return status;
    }
  }
}

