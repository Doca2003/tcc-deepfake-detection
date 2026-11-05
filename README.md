# Detecção de Deepfakes em Imagens com ELA + CNN + Grad-CAM

Trabalho de Conclusão de Curso (Engenharia de Computação).

Estudo de ablação para detecção de deepfakes em **imagens**, avaliando o efeito do
Error Level Analysis (ELA) como pré-processamento e comparando duas arquiteturas de CNN.

## Configurações do estudo de ablação

| Config | Entrada | Arquitetura |
|--------|---------|-------------|
| A | Rostos recortados (sem ELA) | ResNet |
| B | Mapas ELA | ResNet |
| C | Mapas ELA | CNN própria (mais simples) |

- **A → B**: isola o efeito do ELA (mesma arquitetura).
- **B → C**: isola o efeito da arquitetura (mesma entrada).

Avaliação: acurácia, precisão, recall e F1-score, seguida de Grad-CAM e
comparação qualitativa entre os mapas do Grad-CAM e os mapas ELA.

## Pipeline

```
vídeo (.mp4) → extração de frames → detecção/recorte de rosto (MTCNN) → mapa ELA
```

Os frames e rostos são salvos em PNG (sem perdas) para que o ELA seja calculado
com uma única recompressão JPEG controlada (qualidade 90).

## Estrutura do projeto

```
src/
├── preprocessing/
│   ├── extract_frames.py     # vídeo -> frames
│   ├── face_detection.py     # frame -> rosto recortado (224x224)
│   └── ela.py                # rosto -> mapa ELA
├── utils/
│   └── dataset.py            # tf.data + split por vídeo
├── models/                   # (em desenvolvimento)
├── training/                 # (em desenvolvimento)
├── evaluation/               # (em desenvolvimento)
├── run_extraction.py
├── run_face_detection.py
└── run_ela.py
```

## Como executar

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Coloque vídeos em `data/samples/videos/real/` e `data/samples/videos/fake/` e,
a partir da raiz do projeto, rode:

```bash
python src/run_extraction.py
python src/run_face_detection.py
python src/run_ela.py
python src/test_dataset.py     # valida o carregamento dos dados
```

## Decisões de metodologia

- Divisão treino/validação/teste feita **por vídeo** (não por frame), para evitar
  vazamento de dados entre conjuntos.
- Mesma divisão (seed fixa) nas três configurações, garantindo comparação justa.
- Mesmas condições de treino em A, B e C.

## Status

- [x] Extração de frames
- [x] Detecção e recorte de rostos
- [x] Aplicação do ELA
- [x] Carregamento dos dados e split por vídeo
- [x] Modelos (ResNet e CNN própria)
- [x] Treinamento e métricas
- [ ] Grad-CAM e análise qualitativa

## Dados

Os datasets não estão neste repositório. O treino completo é feito no Kaggle.
O FaceForensics++ exige aceite dos termos de uso dos autores.

## Autor

Dominique Guedes
