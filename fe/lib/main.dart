import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'services/api_service.dart';
import 'services/storage_service.dart';
import 'providers/connection_provider.dart';
import 'providers/run_provider.dart';
import 'providers/history_provider.dart';
import 'screens/home_screen.dart';
import 'screens/progress_screen.dart';
import 'screens/results_screen.dart';
import 'screens/history_screen.dart';
import 'screens/settings_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Initialize services
  final storageService = StorageService();
  final serverUrl = await storageService.getServerUrl();
  final apiService = ApiService(baseUrl: serverUrl);
  
  runApp(MyApp(apiService: apiService));
}

class MyApp extends StatelessWidget {
  final ApiService apiService;

  const MyApp({super.key, required this.apiService});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(
          create: (_) => ConnectionProvider(apiService),
        ),
        ChangeNotifierProvider(
          create: (_) => RunProvider(apiService),
        ),
        ChangeNotifierProvider(
          create: (_) => HistoryProvider(apiService),
        ),
      ],
      child: MaterialApp(
        title: 'Rilevatore Bright Spot',
        theme: ThemeData(
          colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
          useMaterial3: true,
        ),
        initialRoute: '/',
        routes: {
          '/': (context) => const HomeScreen(),
          '/progress': (context) => const ProgressScreen(),
          '/results': (context) => const ResultsScreen(),
          '/history': (context) => const HistoryScreen(),
          '/settings': (context) => const SettingsScreen(),
        },
      ),
    );
  }
}
