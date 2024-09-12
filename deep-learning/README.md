# Inteligência Artificial

- [Introdução](#introdução)
- [Perceptron](#perceptron)
- [Multi Layer Perceptron (MLP)](#multi-layer-perceptron-mlp)
- Backpropagation e Regularização
- Keras e TF (Tensorflow)

## Introdução

Este repositório contém materiais e atividades desenvolvidos durante meu estudo de Inteligência Artificial (IA) com foco em Deep Learning. O objetivo é consolidar os conceitos fundamentais e avançados abordados no curso, desde a teoria até a prática.

### Inteligência Artificial
A IA envolve qualquer técnica que permita aos computadores imitar o comportamento humano, seja por meio de lógica, árvores de decisão ou aprendizado de máquina. As áreas abordadas incluem:

- Visão Computacional
- Processamento de Linguagem Natural
- Reconhecimento de Fala
- Robótica

### Aprendizado de Máquina
Um subconjunto da IA, o Aprendizado de Máquina se refere ao uso de técnicas estatísticas que permitem às máquinas melhorar seu desempenho em tarefas com o tempo. Aqui, são incluídos três paradigmas principais:

- Supervisionado
- Não-supervisionado
- Reforço

### Aprendizado Profundo
O Aprendizado Profundo é uma técnica avançada dentro do aprendizado de máquina, caracterizada pelo uso de redes neurais profundas (com várias camadas). Essas redes são capazes de aprender representações hierárquicas dos dados, permitindo a realização de tarefas complexas como reconhecimento de imagens e processamento de linguagem natural.

Marcos Históricos do Aprendizado Profundo:
- **1957**: Introdução do conceito de redes neurais com a publicação de "Perceptrons" por Frank Rosenblatt.
- **1986**: Introdução do algoritmo de backpropagation, um avanço crucial para o treinamento de redes neurais.
- **2012**: Rede neural profunda vence competição de reconhecimento de imagem, liderada pela equipe de Andrew Ng.
- **2015**: AlphaGo, da equipe de Demis Hassabis, vence o jogo de Go, utilizando uma rede neural profunda.
- **2017**: Publicação do artigo "Transformers", que revolucionou o processamento de linguagem natural.
Além disso, o progresso no campo do Aprendizado Profundo foi impulsionado pela disponibilidade de grandes volumes de dados (Big Data), avanços em hardware (como GPUs) e a evolução de frameworks de software como TensorFlow e PyTorch.

## Perceptron

O **Perceptron** é uma das arquiteturas de redes neurais artificiais mais simples, inventada em 1957 por **Frank Rosenblatt**. Ele foi inspirado no funcionamento dos neurônios biológicos, onde neurônios frequentemente acionados simultaneamente reforçam suas conexões. Esse conceito é traduzido para o Perceptron por meio de um modelo matemático que realiza uma combinação linear das entradas e decide uma saída com base em um limiar.

### Neurônio Artificial

O Perceptron é baseado no modelo de **Neurônio Artificial**, inspirado na forma como os neurônios biológicos se comunicam. Um neurônio artificial recebe entradas, pondera essas entradas por meio de pesos e calcula uma saída. O conceito de **TLU (Threshold Logic Unit)** ou **LTU (Linear Threshold Unit)** define esse neurônio como um mecanismo que realiza uma soma ponderada e aplica uma função de ativação degrau, gerando uma saída binária.

### Circuitos Lógicos com Neurônios
Os neurônios artificiais podem ser usados para simular **circuitos lógicos** como AND, OR e NOT. Um neurônio com entradas binárias pode calcular operações lógicas simples, como mostrado na tabela abaixo:

**AND:**

| A | B | A && B |
|---|---|--------|
| 0 | 0 |   0    |
| 0 | 1 |   0    |
| 1 | 0 |   0    |
| 1 | 1 |   1    |

**OR:**
| A | B | A \|\| B |
|---|---|----------|
| 0 | 0 |   0      |
| 0 | 1 |   1      |
| 1 | 0 |   1      |
| 1 | 1 |   1      |

**NOT:**
| A | B | !B | A && !B |
|---|---|----|---------|
| 0 | 0 |  1 |    0    |
| 0 | 1 |  0 |    0    |
| 1 | 0 |  1 |    1    |
| 1 | 1 |  0 |    0    |

Essas redes neurais podem ser combinadas para computar expressões lógicas mais complexas, que são a base de decisões computacionais.

### Forward Propagation
No Perceptron, o cálculo segue o fluxo da **propagação direta**:

- **Entrada → Peso → Soma → Função de Ativação → Saída.**

Esse processo é análogo ao comportamento de um neurônio biológico que decide se "dispara" ou não com base em uma soma ponderada de estímulos.

### Classificação Binária Linear
Um Perceptron é capaz de realizar classificação binária linear simples. Ele calcula uma combinação linear das entradas e, se o resultado exceder um determinado limiar, ele classifica a entrada como pertencente à classe positiva. Caso contrário, classifica na classe negativa.

**Exemplo Prático (AND):**

~~~python
from sklearn.linear_model import Perceptron

perceptron = Perceptron()

X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = [0, 0, 0, 1]

perceptron.fit(X_train, y_train)

X_test = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_pred = perceptron.predict(X_test)

print(y_pred) # output: [0 0 0 1]
~~~

>Neste exemplo, o Perceptron aprende a simular a operação lógica **AND**.

### Limitações do Perceptron
Embora o Perceptron tenha sido um avanço importante, **Marvin Minsky** e **Seymour Papert**, em 1969, destacaram suas limitações em uma monografia. O Perceptron simples não consegue resolver problemas que envolvem **classificação não linear**, como o famoso caso do **OR exclusivo (XOR)**, onde não é possível traçar uma linha reta para separar as classes.

### Problemas do Perceptron:
- Incapacidade de resolver problemas não lineares, como XOR.
- Não gera probabilidades de classe.
- Sua simplicidade limita a aplicação em problemas mais complexos.

Essas limitações podem ser superadas com o uso de arquiteturas mais complexas, como o **Multilayer Perceptron (MLP)**, que utiliza múltiplas camadas de neurônios para lidar com padrões mais complexos e realizar classificações não lineares.

## Multi Layer Perceptron (MLP)

O **Multilayer Perceptron (MLP)** é uma extensão do Perceptron simples e uma das arquiteturas mais fundamentais em redes neurais. Ao contrário do Perceptron de camada única, o MLP possui múltiplas camadas de neurônios, permitindo que ele resolva problemas que envolvem classificação não linear, como o famoso problema do XOR.

### Estrutura do MLP
Um MLP típico é composto por:

- **Uma camada de entrada:** onde os dados brutos são alimentados.
- **Camadas ocultas:** uma ou mais camadas intermediárias onde os neurônios processam os dados com base em pesos e funções de ativação.
- **Camada de saída:** fornece a predição final do modelo, geralmente após aplicar uma função de ativação.

>Além disso, cada camada contém neurônios de viés, que ajudam a ajustar o modelo para melhorar a performance.

### Funcionamento
O MLP utiliza dois processos principais para seu aprendizado:

- **Forward Pass (Propagação Direta):** Os dados são propagados através da rede, da camada de entrada até a camada de saída, onde a predição é gerada.
- **Backward Pass (Retropropagação):** Após calcular o erro da predição (diferença entre a saída prevista e o valor real), o erro é propagado para trás, ajustando os pesos das conexões para melhorar as previsões futuras.

### Funções de Ativação
As funções de ativação desempenham um papel crucial na introdução de não-linearidade, permitindo que a rede aprenda padrões complexos. As mais comuns em MLPs são:

- **Sigmoid (ou logística):** 
``σ(z) = 1 / (1 + exp(-z))``

- **Tanh (Tangente Hiperbólica):** 
``tanh(z) = 2σ(2z) - 1``

- **ReLU (Rectified Linear Unit):** 
``ReLU(z) = max(0, z)``

Outras variações, como Leaky ReLU, ELU (Exponential Linear Unit) e SELU (Scaled ELU), também são usadas para melhorar a performance, especialmente em redes mais profundas.

### Aplicações do MLP
O MLP pode ser utilizado tanto para classificação binária quanto para classificação multiclasse:

- **Classificação Binária:** Usa um neurônio de saída com função de ativação logística (**sigmoid**), onde a saída pode ser interpretada como a probabilidade estimada da classe positiva.
- **Classificação Multiclasse:** Para esse caso, utiliza-se um neurônio de saída por classe e a função de ativação **softmax** para normalizar as saídas e fornecer probabilidades para cada classe.

>Softmax: Converte as saídas de uma rede neural em probabilidades, garantindo que a soma seja 1. Usada em classificação multiclasse.

### Superando as Limitações do Perceptron
Ao adicionar camadas ocultas e funções de ativação não lineares, o MLP é capaz de superar as limitações do Perceptron simples, conseguindo resolver problemas como:
- XOR.
- Reconhecimento de Padrões.
- Regressão.
- Classificação Multiclasse.

### Exemplo prático (XOR)

~~~python
from sklearn.neural_network import MLPClassifier

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]  # Saídas esperadas para XOR

mlp = MLPClassifier(hidden_layer_sizes=(2,), activation='relu', solver='adam', max_iter=1000, random_state=42)

mlp.fit(X, y)

y_pred = mlp.predict(X)

print(f"Predições para XOR: {y_pred}")
~~~

