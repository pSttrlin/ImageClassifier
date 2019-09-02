<h1>Werbung in Zeitungen erkennen</h1>


Tensorflow mit Keras backend. Simples CNN mit 2 Convolutional Layern und 2 Fully Connected Layern
Getestet auf Windows 10 und Python 3.6.0

Dependencies:

    pip install tensorflow==1.14
  
    pip install opencv-python
  
    pip install numpy

<h3>ConfigParser.py</h3>

Parst Argumente

Alle Ordner müssen existieren, und werden nicht erstellt

--config = Optionale config Datei die Argumente beinhaltet (Standart=defaultConfig.cfg)

--num_classes = Anzahl der möglichen ausgaben (2 für Werbung und Nicht Werbung)

--num_conv_layers = Anzahl der convolutional layers im Model

--num_fully_layers = Anzahl der fully connected layers im Model

--model_path = Pfad zum gespeichertem Model das in predict.py geladen werden soll

--model_dir = Ordner in dem models gespeichert werden sollen


--lr = Learning rate

--num_epochs = Anzahl der Epochen

--tensorboard_logdir = Ordner um Tensoboard Dateien zu speichern

--image_width = Breite der Bilder

--image_height = Höhe der Bilder

--batch_size = Batch size zum Trainieren

--num_channels = Anzahl der Farben in Bildern (1 = Schwarz Weiß, 3 = RGB)

--shuffle = Trainingsdaten vermischen?

--testing_path = Pfad zu den Testbildern (Sollte einen Ordner 'Ads' und 'Other' enthalten die dann jeweils .jpeg Bilder mit Werbung und ohne beinhalten (Oder in Sub-Ordnern))

--training_path = Pfad zu den Trainingsbilder (Selbe Struktur wie TESTING_PATH)

--images_per_step = Anzahl der Bilder die gleichzeitig beim Trainieren in den Speicher geladen werden


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

Speichert den Trainingsprozess mit Tensorboard

        tensorboard --logdir logs

Startet tensorboard auf dem Port 6006 (http://localhost:6006),  um den Trainingsprozess in Echtzeit zu verfolgen
