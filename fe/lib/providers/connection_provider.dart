import 'dart:async';
import 'package:flutter/foundation.dart';
import '../services/api_service.dart';

class ConnectionProvider with ChangeNotifier {
  final ApiService _apiService;
  Timer? _healthCheckTimer;
  bool _isConnected = false;
  bool _isChecking = false;

  ConnectionProvider(this._apiService) {
    startHealthCheck();
  }

  bool get isConnected => _isConnected;
  bool get isChecking => _isChecking;

  void startHealthCheck() {
    _healthCheckTimer?.cancel();
    _healthCheckTimer = Timer.periodic(const Duration(seconds: 3), (_) {
      checkConnection();
    });
    // Check immediately
    checkConnection();
  }

  void stopHealthCheck() {
    _healthCheckTimer?.cancel();
    _healthCheckTimer = null;
  }

  Future<void> checkConnection() async {
    if (_isChecking) return;
    
    _isChecking = true;
    notifyListeners();

    try {
      _isConnected = await _apiService.checkHealth();
    } catch (e) {
      _isConnected = false;
    } finally {
      _isChecking = false;
      notifyListeners();
    }
  }

  @override
  void dispose() {
    stopHealthCheck();
    super.dispose();
  }
}

