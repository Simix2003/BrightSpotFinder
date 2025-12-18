# BrightSpot Detector - Flutter App

Applicazione Flutter Windows per il rilevamento di bright spot nelle immagini.

## Requisiti

- Flutter 3.0+
- Windows 10+

## Installazione

1. Installa le dipendenze:
```bash
flutter pub get
```

## Esecuzione

### Modalità Sviluppo

```bash
flutter run -d windows
```

### Build Release

```bash
flutter build windows
```

L'applicazione sarà disponibile in `build/windows/x64/runner/Release/`

## Configurazione

1. Avvia il server Python backend (vedi `../backend/README.md`)
2. Apri l'app Flutter
3. Verifica che lo stato di connessione sia verde
4. Configura i parametri:
   - Seleziona il modello AI (.pt file)
   - Seleziona la cartella input
   - Seleziona la cartella output
   - Imposta la confidenza minima
   - Inserisci un nome per l'elaborazione
5. Clicca "Avvia Elaborazione"

## Funzionalità

- **Home Screen**: Configurazione parametri e avvio elaborazione
- **Progress Screen**: Monitoraggio in tempo reale dell'elaborazione
- **Results Screen**: Visualizzazione statistiche finali
- **History Screen**: Storico delle elaborazioni con statistiche combinate
- **Settings Screen**: Configurazione batch size e porta server

## Note

- L'app deve essere riavviata dopo aver modificato le impostazioni del server
- Il server Python deve essere in esecuzione prima di avviare un'elaborazione
- Le statistiche vengono salvate in `%APPDATA%/BrightSpotDetector/`
