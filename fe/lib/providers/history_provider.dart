import 'package:flutter/foundation.dart';
import '../models/run_model.dart';
import '../services/api_service.dart';

class HistoryProvider with ChangeNotifier {
  final ApiService _apiService;
  
  List<RunModel> _runs = [];
  bool _isLoading = false;
  String? _error;
  Set<String> _selectedRunIds = {};

  HistoryProvider(this._apiService);

  List<RunModel> get runs => _runs;
  bool get isLoading => _isLoading;
  String? get error => _error;
  Set<String> get selectedRunIds => _selectedRunIds;
  bool get hasSelection => _selectedRunIds.isNotEmpty;

  Future<void> loadHistory({int? limit}) async {
    _isLoading = true;
    _error = null;
    notifyListeners();

    try {
      _runs = await _apiService.getHistory(limit: limit);
      _error = null;
    } catch (e) {
      _error = e.toString();
      _runs = [];
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  void toggleRunSelection(String runId) {
    if (_selectedRunIds.contains(runId)) {
      _selectedRunIds.remove(runId);
    } else {
      _selectedRunIds.add(runId);
    }
    notifyListeners();
  }

  void selectAll() {
    _selectedRunIds = _runs.map((r) => r.runId).toSet();
    notifyListeners();
  }

  void deselectAll() {
    _selectedRunIds.clear();
    notifyListeners();
  }

  bool isSelected(String runId) {
    return _selectedRunIds.contains(runId);
  }
}

