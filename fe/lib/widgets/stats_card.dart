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
              Icons.cancel,
              Colors.red,
            ),
            const SizedBox(height: 8),
            _buildStatRow(
              'Senza Bright Spot',
              stats.imagesWithoutBrightSpot.toString(),
              Icons.check_circle,
              Colors.green,
            ),
            const SizedBox(height: 8),
            _buildStatRow(
              'Tasso di Successo',
              '${stats.successRate.toStringAsFixed(1)}%',
              Icons.trending_up,
              Colors.blue,
            ),
            const SizedBox(height: 8),
            _buildStatRow(
              'Tasso di Bright Spot',
              '${stats.brightSpotRate.toStringAsFixed(1)}%',
              Icons.warning,
              Colors.orange,
            ),
            if (stats.startTime != null) ...[
              const SizedBox(height: 8),
              _buildStatRow(
                'Ora Inizio',
                _formatStartTime(stats.startTime!),
                Icons.access_time,
              ),
            ],
            if (stats.startTime != null) ...[
              const SizedBox(height: 8),
              _buildStatRow(
                'Tempo Trascorso',
                _formatElapsedTime(stats.startTime!),
                Icons.timer,
                Colors.blue,
              ),
            ],
            if (stats.medianInferenceTimeSeconds != null) ...[
              const SizedBox(height: 8),
              _buildStatRow(
                'Tempo Mediano per Immagine',
                _formatDuration(stats.medianInferenceTimeSeconds!),
                Icons.speed,
                Colors.purple,
              ),
            ],
            if (stats.durationSeconds != null) ...[
              const SizedBox(height: 8),
              _buildStatRow(
                'Durata Totale',
                _formatDuration(stats.durationSeconds!),
                Icons.schedule,
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
    final duration = Duration(milliseconds: (seconds * 1000).round());
    final hours = duration.inHours;
    final minutes = duration.inMinutes.remainder(60);
    final secs = duration.inSeconds.remainder(60);
    final millis = duration.inMilliseconds.remainder(1000);

    if (hours > 0) {
      return '${hours}h ${minutes}m ${secs}s';
    } else if (minutes > 0) {
      return '${minutes}m ${secs}s';
    } else if (secs > 0) {
      return '${secs}s';
    } else {
      return '${millis}ms';
    }
  }

  String _formatStartTime(DateTime startTime) {
    final hour = startTime.hour.toString().padLeft(2, '0');
    final minute = startTime.minute.toString().padLeft(2, '0');
    final second = startTime.second.toString().padLeft(2, '0');
    return '$hour:$minute:$second';
  }

  String _formatElapsedTime(DateTime startTime) {
    final now = DateTime.now();
    final elapsed = now.difference(startTime);
    final hours = elapsed.inHours;
    final minutes = elapsed.inMinutes.remainder(60);
    final secs = elapsed.inSeconds.remainder(60);

    if (hours > 0) {
      return '${hours}h ${minutes}m ${secs}s';
    } else if (minutes > 0) {
      return '${minutes}m ${secs}s';
    } else {
      return '${secs}s';
    }
  }
}

