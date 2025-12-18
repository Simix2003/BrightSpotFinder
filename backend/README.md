# BrightSpot Detector - Backend Server

Server Python REST API per il rilevamento di bright spot nelle immagini utilizzando YOLO.

## Requisiti

- Python 3.8+
- Windows 10+

## Installazione

1. Installa le dipendenze:
```bash
pip install -r requirements.txt
```

## Utilizzo

### Esecuzione come Script Python

```bash
python server.py
```

Il server sarà disponibile su `http://localhost:5000`

### Compilazione in EXE

Per creare un eseguibile standalone:

```bash
python build_exe.py
```

L'EXE sarà creato nella cartella `dist/` con il nome `BrightSpotDetectorServer.exe`

**Nota**: L'EXE sarà di dimensioni considerevoli (circa 500MB-1GB) a causa delle dipendenze PyTorch e YOLO.

## API Endpoints

- `GET /health` - Verifica connessione server
- `POST /api/start` - Avvia elaborazione
- `GET /api/status/<run_id>` - Stato elaborazione
- `GET /api/stats/<run_id>` - Statistiche run completata
- `GET /api/history` - Lista tutte le run
- `POST /api/stats/combined` - Statistiche combinate per multiple run
- `GET /api/batch/status/<run_id>` - Stato dettagliato batch

## Persistenza Dati

I dati vengono salvati in `%APPDATA%/BrightSpotDetector/`:
- `runs/` - Dati delle run completate
- `checkpoints/` - Checkpoint per resume automatico

Le run più vecchie di 1 anno vengono eliminate automaticamente all'avvio.

## Batch Processing

Il server divide automaticamente le immagini in batch della dimensione specificata. Dopo ogni batch completato, viene salvato un checkpoint per permettere il resume automatico in caso di crash.

