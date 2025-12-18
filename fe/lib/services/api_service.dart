import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/run_model.dart';
import '../models/stats_model.dart';

class ApiService {
  final String baseUrl;
  final Duration timeout;

  ApiService({String? baseUrl, Duration? timeout})
      : baseUrl = baseUrl ?? 'http://localhost:5000',
        timeout = timeout ?? const Duration(seconds: 30);

  Future<bool> checkHealth() async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/health'))
          .timeout(timeout);
      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }

  Future<String> startProcessing({
    required String modelPath,
    required String inputDir,
    required String outputDir,
    required double confidence,
    required String runName,
    required int batchSize,
  }) async {
    final response = await http
        .post(
          Uri.parse('$baseUrl/api/start'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({
            'model_path': modelPath,
            'input_dir': inputDir,
            'output_dir': outputDir,
            'confidence': confidence,
            'run_name': runName,
            'batch_size': batchSize,
          }),
        )
        .timeout(timeout);

    if (response.statusCode != 200) {
      final error = jsonDecode(response.body)['error'] as String?;
      throw Exception(error ?? 'Failed to start processing');
    }

    final data = jsonDecode(response.body) as Map<String, dynamic>;
    return data['run_id'] as String;
  }

  Future<StatusResponse> getStatus(String runId) async {
    final response = await http
        .get(Uri.parse('$baseUrl/api/status/$runId'))
        .timeout(timeout);

    if (response.statusCode != 200) {
      try {
        final errorData = jsonDecode(response.body) as Map<String, dynamic>;
        final error = errorData['error'] as String?;
        throw Exception(error ?? 'Failed to get status');
      } catch (e) {
        throw Exception('Failed to get status: ${response.statusCode}');
      }
    }

    try {
      final data = jsonDecode(response.body) as Map<String, dynamic>;
      return StatusResponse.fromJson(data);
    } catch (e) {
      throw Exception('Failed to parse status response: $e');
    }
  }

  Future<RunModel> getStats(String runId) async {
    final response = await http
        .get(Uri.parse('$baseUrl/api/stats/$runId'))
        .timeout(timeout);

    if (response.statusCode != 200) {
      final error = jsonDecode(response.body)['error'] as String?;
      throw Exception(error ?? 'Failed to get stats');
    }

    final data = jsonDecode(response.body) as Map<String, dynamic>;
    return RunModel.fromJson(data);
  }

  Future<List<RunModel>> getHistory({int? limit}) async {
    final uri = limit != null
        ? Uri.parse('$baseUrl/api/history?limit=$limit')
        : Uri.parse('$baseUrl/api/history');

    final response = await http.get(uri).timeout(timeout);

    if (response.statusCode != 200) {
      final error = jsonDecode(response.body)['error'] as String?;
      throw Exception(error ?? 'Failed to get history');
    }

    final data = jsonDecode(response.body) as Map<String, dynamic>;
    final runs = data['runs'] as List;
    return runs.map((r) => RunModel.fromJson(r as Map<String, dynamic>)).toList();
  }

  Future<CombinedStatsResponse> getCombinedStats(List<String> runIds) async {
    final response = await http
        .post(
          Uri.parse('$baseUrl/api/stats/combined'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({'run_ids': runIds}),
        )
        .timeout(timeout);

    if (response.statusCode != 200) {
      final error = jsonDecode(response.body)['error'] as String?;
      throw Exception(error ?? 'Failed to get combined stats');
    }

    final data = jsonDecode(response.body) as Map<String, dynamic>;
    return CombinedStatsResponse.fromJson(data);
  }
}

