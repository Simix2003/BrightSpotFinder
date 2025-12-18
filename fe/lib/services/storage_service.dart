import 'package:shared_preferences/shared_preferences.dart';

class StorageService {
  static const String _keyServerPort = 'server_port';
  static const String _keyBatchSize = 'batch_size';
  static const String _keyServerUrl = 'server_url';

  Future<void> saveServerPort(int port) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt(_keyServerPort, port);
  }

  Future<int> getServerPort() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getInt(_keyServerPort) ?? 5000;
  }

  Future<void> saveBatchSize(int batchSize) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt(_keyBatchSize, batchSize);
  }

  Future<int> getBatchSize() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getInt(_keyBatchSize) ?? 100;
  }

  Future<void> saveServerUrl(String url) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_keyServerUrl, url);
  }

  Future<String> getServerUrl() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_keyServerUrl) ?? 'http://localhost:5000';
  }
}

