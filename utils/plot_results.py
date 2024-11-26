import matplotlib.pyplot as plt
import seaborn as sns

# Dados
dados = {
    'RandomSampling': {'data': [0.5223171889838556, 0.4358974358974359, 0.5356125356125356, 0.5004748338081672, 0.5161443494776828, 0.4919278252611586, 0.47910731244064575, 0.5208926875593543, 0.5246913580246914, 0.4843304843304843, 0.5479582146248813]},
    'MarginSampling': {'data': [0.5223171889838556, 0.6324786324786325, 0.5018993352326686, 0.5593542260208927, 0.5944919278252612, 0.7288698955365622, 0.6438746438746439, 0.5698005698005698, 0.4453941120607787, 0.5246913580246914, 0.5370370370370371]},
    'EntropySampling': {'data': [0.5223171889838556, 0.5090218423551757, 0.47768281101614435, 0.6562203228869895, 0.6452991452991453, 0.6419753086419753, 0.6163342830009497, 0.6220322886989553, 0.5546058879392213, 0.6286799620132953, 0.7022792022792023]},
    'LeastConfidence': {'data': [0.5223171889838556, 0.6533713200379867, 0.4990503323836657, 0.6215574548907882, 0.5716999050332384, 0.6723646723646723, 0.5821462488129154, 0.6752136752136753, 0.5142450142450142, 0.6020892687559354, 0.6481481481481481]},
    #'MODEL': {'data': [0.4, 0.54, 0.85]},
    #'MODEL': {'data': [0.4, 0.54, 0.85]},
    #'MODEL': {'data': [0.4, 0.54, 0.85]},
    #'MODEL': {'data': [0.4, 0.54, 0.85]},
}
local_rounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Configuração do estilo com seaborn
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))

# Plotando os dados
for modelo, valores in dados.items():
    plt.plot(local_rounds, valores['data'], marker='o', label=modelo)

# Configurações do gráfico
plt.title("Model comparison", fontsize=14)
plt.xlabel("Rounds", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.legend(title="Models")
plt.tight_layout()

# Salvando o gráfico
plt.savefig("grafico_10_epochs_final.pdf")
plt.show()
