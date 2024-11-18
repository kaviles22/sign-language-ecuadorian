# sign-language-detector-python

## Fork de [computervisioneng/sign-language-detector-python](https://github.com/computervisioneng/sign-language-detector-python)

Este repositorio es un fork del proyecto [computervisioneng/sign-language-detector-python](https://github.com/computervisioneng/sign-language-detector-python). He realizado modificaciones sobre la versión original para detectar las letras del abecedario del lenguaje de senas ecuatoriano. Adicionalmente se anadió una parte de detección de palabras, usando detección de acciones. Este repo busca tener una app lista para usar.


### Detección de letras:
#### Uso: detección de landmarks y clasificación
Esta funcionalidad clasifica frame a frame los landmarks detectados previamente por la librería mediapipe, no toma en cuenta la secuencialidad de puntos.

### Limitaciones:
- En la parte de clasificación NO se detectan las letras con moviemiento como la j, n, rr, ll.

### Detección de palabras
#### Uso: detección de acciones
Esta funcionalidad clasifica palabras, usando ventanas de tiempo para detectar acciones.