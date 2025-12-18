import 'dart:async';
import 'package:flutter/foundation.dart';
import '../models/stats_model.dart';
import '../services/api_service.dart';

class RunProvider with ChangeNotifier {
  final ApiService _apiService;
  Timer? _statusTimer;
  
  String? _currentRunId;
  StatusResponse? _currentStatus;
  bool _isProcessing = false;
  String? _error;

  RunProvider(this._apiService);

  String? get currentRunId => _currentRunId;
  StatusResponse? get currentStatus => _currentStatus;
  bool get isProcessing => _isProcessing;
  String? get error => _error;

  Future<String> startRun({
    required String modelPath,
    required String inputDir,
    required String outputDir,
    required double confidence,
    required String runName,
    required int batchSize,
  }) async {
    try {
      _error = null;
      _isProcessing = true;
      notifyListeners();

      _currentRunId = await _apiService.startProcessing(
        modelPath: modelPath,
        inputDir: inputDir,
        outputDir: outputDir,
        confidence: confidence,
        runName: runName,
        batchSize: batchSize,
      );

      // Start polling for status
      startStatusPolling();
      
      notifyListeners();
      return _currentRunId!;
    } catch (e) {
      _error = e.toString();
      _isProcessing = false;
      notifyListeners();
      rethrow;
    }
  }

  void startStatusPolling() {
    _statusTimer?.cancel();
    _statusTimer = Timer.periodic(const Duration(seconds: 2), (_) {
      if (_currentRunId != null) {
        _updateStatus();
      }
    });
    // Update immediately
    _updateStatus();
  }

  void stopStatusPolling() {
    _statusTimer?.cancel();
    _statusTimer = null;
  }

  Future<void> _updateStatus() async {
    if (_currentRunId == null) return;

    try {
      final status = await _apiService.getStatus(_currentRunId!);
      _currentStatus = status;
      
      // Check if completed or failed
      if (status.status == 'completed' || status.status == 'failed') {
        stopStatusPolling();
        _isProcessing = false;
      }
      
      notifyListeners();
    } catch (e) {
      // Don't update error on polling failures, just log
      debugPrint('Status polling error: $e');
    }
  }

  void reset() {
    stopStatusPolling();
    _currentRunId = null;
    _currentStatus = null;
    _isProcessing = false;
    _error = null;
    notifyListeners();
  }

  @override
  void dispose() {
    stopStatusPolling();
    super.dispose();
  }
}

