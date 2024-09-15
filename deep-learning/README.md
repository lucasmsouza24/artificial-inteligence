# Inteligência Artificial

- [Introdução](#introdução)
- [Perceptron](#perceptron)
- [Multi Layer Perceptron (MLP)](#multi-layer-perceptron-mlp)
- [Backpropagation e Regularização](#backpropagation-e-regularização)
- [Keras e TF (Tensorflow)](#keras-e-tf-tensorflow)

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

#### O que é uma Função de Ativação?
As funções de ativação são elementos fundamentais em redes neurais. Elas são usadas para definir se um neurônio deve ser "ativado" ou não, e introduzem não-linearidade no modelo. Isso permite que a rede aprenda e represente relações complexas e não lineares nos dados, essenciais para resolver problemas mais desafiadores, como o reconhecimento de padrões e a classificação de imagens.

#### Por que precisamos de funções de ativação?
Sem funções de ativação, as redes neurais funcionariam como um simples modelo linear, incapaz de capturar padrões complexos. As funções de ativação possibilitam que a rede aprenda e resolva problemas de classificação não linear, como o **XOR**.

#### Principais Funções de Ativação
Agora, vamos explorar as principais funções de ativação citadas: **Sigmoid**, **Tanh**, **ReLU** e **Softmax**, destacando suas características e quando elas são mais indicadas.

**1. Sigmoid (Logística)**
Fórmula: ``σ(z) = 1 / (1 + exp(-z))``

>**Como funciona:** Mapeia qualquer valor de entrada para um intervalo entre **0 e 1**, tornando-a útil para tarefas de classificação binária. Se a saída for maior que 0.5, pode-se considerar como pertencente à classe positiva.

**Pontos importantes:**

- **Suavidade:** Produz uma transição suave entre as classes.
- **Uso:** Comumente utilizada para saídas binárias em problemas de classificação.
- **Problema:** Pode sofrer com o problema do desvanecimento do gradiente em redes profundas, o que dificulta o treinamento.

**2. Tanh (Tangente Hiperbólica)**
Fórmula: ``tanh(z) = 2σ(2z) - 1``

>**Como funciona:** Assim como a sigmoid, mapeia valores, mas em um intervalo de **-1 a 1**, o que a torna mais centralizada em torno de zero. Isso facilita a convergência durante o treinamento.

**Pontos importantes:**

- **Centro em zero:** Ao contrário da sigmoid, o valor de saída de neurônios **pode ser negativo**, o que ajuda a modelar dados que tenham valores distribuídos de forma positiva e negativa.
- **Uso:** Boa escolha para camadas ocultas de redes neurais.
- **Problema:** Também pode sofrer com o desvanecimento do gradiente em redes profundas.

**3. ReLU (Rectified Linear Unit)**
Fórmula: ``ReLU(z) = max(0, z)``

>**Como funciona:** Simplesmente transforma entradas negativas em zero, mantendo as entradas positivas inalteradas. ReLU é muito eficiente em termos de cálculo e é amplamente usada em redes neurais profundas.

**Pontos importantes:**

- **Eficiência:** ReLU é computacionalmente leve e ajuda a evitar o problema do desvanecimento do gradiente, sendo preferida em redes profundas.
- **Problema:** Pode sofrer do problema de neurônios mortos, quando muitos valores de entrada são negativos, resultando em um gradiente zero e um neurônio que não "dispara".

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

## Backpropagation e Regularização

### Backpropagation

Backpropagation (ou retropropagação) é o algoritmo usado para ajustar os pesos em uma rede neural durante o processo de treinamento, a fim de minimizar o erro entre o **valor real** e o **valor previsto**.

#### Funcionamento
O processo ocorre em duas fases:

- **Forward Pass (Propagação Direta):** Os dados são passados pelas camadas da rede até a camada de saída, onde uma predição é feita.

- **Backward Pass (Retropropagação):** O erro entre a predição e o valor real **y** é calculado usando uma função de custo (como a **Cross Entropy** para classificação). O erro é então propagado de volta pela rede, ajustando os pesos com base nos **gradientes**. Isso é feito usando o método de descida do gradiente, que busca minimizar o erro ajustando os pesos na direção oposta ao gradiente do erro.

#### Exemplo de função de custo:

- Para problemas de classificação, usa-se frequentemente a **Cross Entropy Loss**:
~~~scss
L(y, ŷ) = - ∑ y log(ŷ)
~~~
- Para regressão, a função de custo comum é o **Erro Quadrático Médio (Mean Squared Error):**
~~~scss
L(y, ŷ) = (1/n) ∑ (y - ŷ)^2
~~~

>Problemas como o **desaparecimento** e **explosão de gradientes** podem ocorrer durante o treinamento, especialmente em redes profundas. Para mitigar esses problemas, são usadas técnicas como a **ReLU** e variações como **Leaky ReLU**.

### Regularização

**Regularização** é uma técnica utilizada para melhorar a capacidade de generalização de uma rede neural, evitando o **overfitting**. Overfitting ocorre quando a rede aprende detalhes específicos dos dados de treino, incluindo o ruído, em vez de aprender padrões generalizáveis.

#### Técnicas de Regularização:

- **Regularização L2 (Ridge):** Penaliza os pesos grandes, adicionando um termo à função de custo que aumenta à medida que os pesos se tornam grandes. Isso força a rede a manter os pesos pequenos, prevenindo overfitting.
- **Regularização L1 (Lasso):** Funciona de maneira semelhante ao L2, mas penaliza o valor absoluto dos pesos. Isso pode levar à eliminação de alguns pesos, promovendo sparsidade.
- **Dropout:** Durante o treinamento, desativa aleatoriamente uma porcentagem de neurônios, forçando a rede a não depender excessivamente de neurônios específicos. Isso ajuda a combater o overfitting e a criar uma rede mais robusta.

#### Quando usar
- **L2** é amplamente utilizado para manter os pesos sob controle e é a escolha padrão em muitos casos.
- **L1** é útil quando se deseja uma solução mais simples, com alguns pesos efetivamente eliminados.
- **Dropout** é popular em redes profundas e pode ser aplicado a várias camadas, forçando a rede a aprender representações mais generalizáveis.

## Keras e TF (Tensorflow)

### TensorFlow e Keras
O **TensorFlow** é uma plataforma de aprendizado profundo de código aberto, desenvolvida pelo **Google**, que permite construir, treinar e implementar modelos de machine learning e deep learning em escala. O **Keras**, por sua vez, é uma API de alto nível construída sobre o TensorFlow, que facilita a criação e o treinamento de redes neurais de maneira simples e rápida.

### O que é o TensorFlow?
O **TensorFlow** é uma biblioteca poderosa que suporta a construção de gráficos computacionais dinâmicos, manipulando dados na forma de tensores. Esses tensores são generalizações de **vetores e matrizes**, representando dados em múltiplas dimensões, que são fundamentais para operações de álgebra linear em aprendizado profundo. Além disso, o TensorFlow facilita o uso de aceleradores de hardware, como **GPUs** e **TPUs**, para otimizar o desempenho em tarefas intensivas de computação.

### O que é Keras?
O **Keras** é uma API de alto nível para redes neurais, projetada para ser fácil de usar e ao mesmo tempo poderosa. Ela foi incorporada ao **TensorFlow** como sua API padrão, chamada de **tf.keras**, e fornece diversas funcionalidades para construir, compilar e treinar redes neurais complexas com apenas algumas linhas de código.

### Ciclo de Vida de um Modelo em Keras
O ciclo de vida de um modelo em Keras segue as seguintes etapas principais:

1. **Preprocessing():** Pré-processamento dos dados.
2. **Model():** Definição da arquitetura do modelo.
3. **Compile():** Compilação do modelo, especificando a função de custo, otimizador e métricas.
4. **Fit():** Treinamento do modelo com os dados de treino.
5. **Evaluate():** Avaliação do modelo com dados de teste ou validação.
6. **Predict():** Geração de previsões usando o modelo treinado​

### Principais APIs do Keras
O Keras fornece várias APIs e módulos essenciais para a construção e o treinamento de redes neurais:

- ``tf.keras.Sequential:`` Para criar modelos sequenciais, onde as camadas são empilhadas linearmente.
- ``tf.keras.Model:`` Permite criar modelos mais complexos usando a API Funcional, suportando múltiplas entradas e saídas.
- ``tf.keras.layers:`` Fornece uma ampla variedade de camadas como Dense, Conv2D, e LSTM, que são blocos fundamentais para construir redes neurais.
- ``tf.keras.optimizers:`` Inclui otimizadores como SGD, Adam, e RMSprop para ajustar os pesos durante o treinamento.
- ``tf.keras.losses:`` Contém funções de perda como Mean Squared Error e Categorical Crossentropy.
- ``tf.keras.metrics:`` Fornece métricas para avaliar a performance do modelo, como accuracy, precision, e recall.
- ``tf.keras.callbacks:`` Permite executar ações durante o treinamento, como salvar checkpoints e interromper o treinamento antecipadamente com EarlyStopping

### Exemplo de Implementação com Keras
Abaixo, um exemplo de como criar um MLP usando a API Sequential do Keras:

~~~python
import tensorflow as tf
from tensorflow import keras

# Definindo o modelo
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# Compilando o modelo
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# Treinando o modelo
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
~~~