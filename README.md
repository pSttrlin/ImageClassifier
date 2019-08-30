<h1>Werbung in Zeitungen erkennen</h1>

Tensorflow mit Keras backend. Simples CNN mit 2 Convolutional Layern und 2 Fully Connected Layern

<h3>config.py</h3>

Enthält Konstanten für das Model, zum Lernen, und für die Trainings- / Testdaten

Alle Ordner müssen existieren, und werden nicht erstellt

NUM_CLASSES = Anzahl der möglichen ausgaben (2 für Werbung und Nicht Werbung)

NUM_CONV_LAYERS = Anzahl der convolutional layers im Model

NUM_FULLY_LAYERS = Anzahl der fully connected layers im Model

MODEL_PATH = Pfad zum gespeichertem Model das in predict.py geladen werden soll

MODEL_DIR = Ordner in dem models gespeichert werden sollen


LR = Learning rate

NUM_EPOCHS = Anzahl der Epochen

TENSORBOARD_LOGDIR = Ordner um Tensoboard Dateien zu speichern


IMAGE_SIZE = Größe der Bilder

BATCH_SIZE = Batch size zum Trainieren

NUM_CHANNELS = Anzahl der Farben in Bildern (1 = Schwarz Weiß, 3 = RGB)

SHUFFLE = Trainingsdaten vermischen?

TESTING_PATH = Pfad zu den Testbildern (Sollte einen Ordner 'Ads' und 'Other' enthalten die dann jeweils .jpeg Bilder mit Werbung und ohne beinhalten (Oder in Sub-Ordnern))

TRAINING_PATH = Pfad zu den Trainingsbilder (Selbe Struktur wie TESTING_PATH)

IMAGES_PER_STEP = Anzahl der Bilder die gleichzeitig beim Trainieren in den Speicher geladen werden


<h3>Dataset.py</h3>

Lädt die Daten aus TESTING_PATH und TRAINING_PATH und bereitet sie auf

DATA_TRAIN = Lädt nur Trainingsdaten

DATA_TEST = Lädt nur Testdaten

DATA_BOTH = Lädt alle Daten


<h3>model.py</h3>

Hilfsfunktionen zum erstellen des Models


<h3>predict.py</h3>

Lädt Testdaten und sagt jedes Bild voraus, gibt am Ende die Trefferquote aus


<h3>train.py</h3>

Erstellt ein neues Model und trainiert dieses mit allen Trainingsdaten

Speichert alle {IMAGES_PER_STEP} Bilder das Model

