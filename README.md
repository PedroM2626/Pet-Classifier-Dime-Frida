# 🐶🐱 Classificador de Pets: Dime e Frida

Um projeto de classificação de imagens usando Deep Learning para identificar meus queridos pets: **Dime**, meu cachorro que faleceu em 2021 e está para sempre em meu coração, e **Frida**, minha gata que ainda me faz companhia.

<div align="center">
  
  [![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
  [![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)](https://keras.io/)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📖 Sobre o Projeto

Este projeto nasceu da vontade de eternizar a memória do Dime e celebrar a presença da Frida através da tecnologia. Utilizando técnicas de Deep Learning e Transfer Learning com a arquitetura VGG16, o classificador consegue distinguir entre imagens dos meus dois pets preciosos.

**Dime** 🐶 (em memória - 2021)  
**Frida** 🐱 (ainda conosco)

O modelo foi treinado com carinho usando 22 imagens de cada pet, aplicando Transfer Learning para obter resultados precisos mesmo com um dataset pequeno e pessoal.

---

## 🎯 Objetivo

Desenvolver um classificador binário de imagens capaz de identificar automaticamente se uma foto é do Dime ou da Frida, preservando suas memórias através da inteligência artificial.

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.x** - Linguagem de programação principal
- **TensorFlow 2.x** - Framework de Deep Learning
- **Keras** - API de alto nível para redes neurais
- **VGG16** - Arquitetura de rede neural convolucional pré-treinada
- **NumPy** - Computação numérica
- **Matplotlib** - Visualização de dados
- **OpenCV (cv2)** - Processamento de imagens
- **scikit-learn** - Ferramentas de Machine Learning

---

## 📁 Estrutura do Projeto

```
Pet-Classifier-Dime-Frida/
│
├── my_precious-dataset/
│   ├── Dime/                  # 22 imagens do Dime
│   ├── Frida/                 # 22 imagens da Frida
│   ├── tests/                 # Imagens para testar o modelo
│   └── results/               # Resultados das predições
│
├── dime_frida_classifier.py   # Script Python principal
├── dime_frida_classifier.ipynb # Notebook Jupyter
├── debug_log.txt              # Log de execução
└── README.md                  # Este arquivo
```

---

## 📊 Dataset

O dataset é composto por:
- **44 imagens no total**
  - 22 imagens do Dime
  - 22 imagens da Frida
- **Divisão dos dados:**
  - 70% para treinamento
  - 15% para validação
  - 15% para teste
- **Formato:** JPG, JPEG, PNG
- **Pré-processamento:** Redimensionamento para 224x224 pixels

---

## 🚀 Como Usar

### Pré-requisitos

```bash
# Python 3.x instalado
# pip instalado
```

### Instalação

1. Clone o repositório:
```bash
git clone https://github.com/PedroM2626/Pet-Classifier-Dime-Frida.git
cd Pet-Classifier-Dime-Frida
```

2. Instale as dependências:
```bash
pip install tensorflow keras numpy matplotlib opencv-python scikit-learn
```

### Executando o Projeto

#### Opção 1: Script Python

```bash
python dime_frida_classifier.py
```

#### Opção 2: Jupyter Notebook

```bash
jupyter notebook dime_frida_classifier.ipynb
```

**Nota:** Você precisará ajustar o caminho da pasta `root` no código para o caminho correto do seu dataset:
```python
root = r'caminho/para/seu/my_precious-dataset'
```

### Testando o Modelo

1. Coloque suas imagens de teste na pasta `my_precious-dataset/tests/`
2. Execute o script ou notebook
3. Os resultados serão salvos na pasta `my_precious-dataset/results/`

---

## 🧠 Arquitetura do Modelo

O projeto utiliza **Transfer Learning** com a arquitetura **VGG16**:

- **Modelo Base:** VGG16 pré-treinado no ImageNet
- **Estratégia:** 
  - Congelamento de todas as camadas convolucionais
  - Substituição da camada de classificação final
  - Nova camada Dense com 2 neurônios (Dime e Frida) e ativação softmax
- **Otimizador:** Adam
- **Função de Perda:** Categorical Crossentropy
- **Métrica:** Accuracy
- **Treinamento:** 10 épocas com batch size de 128

### Por que VGG16?

VGG16 é uma arquitetura robusta e bem estabelecida que:
- Possui 16 camadas de peso
- Foi treinada em milhões de imagens (ImageNet)
- Apresenta excelente capacidade de extração de features
- É ideal para Transfer Learning com datasets pequenos

---

## 📈 Resultados

O modelo é capaz de classificar corretamente imagens do Dime e da Frida com alta precisão. Os resultados das predições incluem:

- Imagem original
- Classe predita (Dime ou Frida)
- Visualizações salvas automaticamente na pasta `results/`

---

## 🔍 Funcionalidades

- ✅ Carregamento automático de imagens
- ✅ Pré-processamento e normalização
- ✅ Divisão automática em conjuntos de treino/validação/teste
- ✅ Transfer Learning com VGG16
- ✅ Avaliação de desempenho
- ✅ Predição em novas imagens
- ✅ Salvamento automático de resultados
- ✅ Sistema de logging para debug

---

## 📝 Notas Importantes

- O modelo funciona melhor com imagens claras e bem iluminadas dos pets
- Imagens devem estar em formato JPG, JPEG ou PNG
- O tamanho das imagens é automaticamente ajustado para 224x224 pixels
- Recomenda-se uso de GPU para treinamento mais rápido (opcional)

---

## 🤝 Contribuições

Este é um projeto pessoal e emocional, mas sugestões e melhorias são bem-vindas! Sinta-se livre para:

- Abrir issues
- Propor melhorias
- Fazer fork do projeto
- Enviar pull requests

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

---

## 👤 Autor

**Pedro M2626**

- GitHub: [@PedroM2626](https://github.com/PedroM2626)

---

## 💝 Dedicatória

*Este projeto é dedicado à memória do Dime, que foi um companheiro fiel e está para sempre em meu coração, e à Frida, que continua trazendo alegria e amor aos meus dias.*

---

## 📚 Referências

- [VGG16 Paper](https://arxiv.org/abs/1409.1556)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Applications](https://keras.io/api/applications/)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)

---

<div align="center">
  
  **Feito com ❤️ e 🐾**
  
  *Em memória do Dime (2021) e para a Frida*

</div>
