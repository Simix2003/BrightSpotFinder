import 'package:flutter/material.dart';
import '../models/run_model.dart';

class StatsCard extends StatelessWidget {
  final RunStats stats;
  final String? title;

  const StatsCard({
    super.key,
    required this.stats,
    this.title,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 4,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (title != null) ...[
              Text(
                title!,
                style: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const Divider(),
            ],
            _buildStatRow(
              'Immagini Totali',
              stats.totalImages.toString(),
              Icons.image,
            ),
            const SizedBox(height: 8),
            _buildStatRow(
              'Con Bright Spot',
              stats.imagesWithBrightSpot.toString(),
              Icons.check_circle,
              Colors.green,
            ),
            const SizedBox(height: 8),
            _buildStatRow(
              'Senza Bright Spot',
              stats.imagesWithoutBrightSpot.toString(),
              Icons.cancel,
              Colors.grey,
            ),
            const SizedBox(height: 8),
            _buildStatRow(
              'Tasso di Successo',
              '${stats.successRate.toStringAsFixed(1)}%',
              Icons.trending_up,
              Colors.blue,
            ),
            if (stats.durationSeconds != null) ...[
              const SizedBox(height: 8),
              _buildStatRow(
                'Durata',
                _formatDuration(stats.durationSeconds!),
                Icons.timer,
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildStatRow(String label, String value, IconData icon, [Color? color]) {
    return Row(
      children: [
        Icon(icon, color: color, size: 20),
        const SizedBox(width: 8),
        Expanded(
          child: Text(
            label,
            style: const TextStyle(fontSize: 14),
          ),
        ),
        Text(
          value,
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
            color: color,
          ),
        ),
      ],
    );
  }

  String _formatDuration(double seconds) {
    final duration = Duration(seconds: seconds.toInt());
    final hours = duration.inHours;
    final minutes = duration.inMinutes.remainder(60);
    final secs = duration.inSeconds.remainder(60);

    if (hours > 0) {
      return '${hours}h ${minutes}m ${secs}s';
    } else if (minutes > 0) {
      return '${minutes}m ${secs}s';
    } else {
      return '${secs}s';
    }
  }
}

