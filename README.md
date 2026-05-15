```markdown
# VAR do Dominó

Reconhecimento automático de pedras de dominó em tempo real usando visão computacional,
transmitindo o estado da partida para um placar virtual acessível pelo navegador.

## Como funciona

Uma câmera (celular via IP Webcam ou webcam USB) filma a mesa de dominó de cima.
O Python processa o vídeo quadro a quadro com OpenCV, detecta e identifica cada pedra,
e serve o estado do jogo via Flask para uma interface web que exibe o placar ao vivo.

## Dependências

Recomendado instalar pelos pacotes do sistema (especialmente em CPUs mais antigas,
versões recentes do NumPy via pip podem não ser compatíveis com hardware legado):

**Debian/Ubuntu/Raspberry Pi:**
```bash
sudo apt install python3 python3-numpy python3-opencv python3-flask
```

**Arch Linux:**
```bash
sudo pacman -S python python-numpy python-opencv python-flask
```

**Fedora/RHEL:**
```bash
sudo dnf install python3 python3-numpy python3-opencv python3-flask
```

> Se preferir usar pip em ambiente virtual:
> ```bash
> python3 -m venv venv
> source venv/bin/activate
> pip install opencv-python numpy flask
> ```
> Atenção: nesse caso verifique a compatibilidade do NumPy com seu hardware.

## Como usar

### app.py — Servidor principal (vídeo ao vivo)

Processa vídeo em tempo real e serve a interface web do placar.

```bash
# Uso básico — inicia o servidor na porta 5000
python3 app.py

# Modo debug — exibe janelas com máscaras e contornos detectados
python3 app.py -d
```

Após iniciar, acesse no navegador: `http://localhost:5000`

Para usar com celular via IP Webcam, configure o endereço da câmera
nas configurações do app antes de iniciar.

**Fluxo típico de uso:**

1. Posicione a câmera apontada para a mesa de cima, em modo paisagem
2. Inicie o servidor com `python3 app.py`
3. Abra a interface no navegador
4. Clique em **[+] Escanear** no jogador correspondente para registrar a mão
5. Force a leitura da mesa com **Forçar Mesa** após as jogadas

---

### baseApp.py — Ferramenta de desenvolvimento e calibração

Processa uma imagem estática (foto da mesa) em vez de vídeo ao vivo.
Útil para calibrar parâmetros como `area_ponto` e depurar o reconhecimento
sem precisar da câmera conectada.

```bash
# Processa uma imagem e exibe o resultado
python3 baseApp.py imagem.jpg

# Modo debug — exibe cada etapa: máscara, metades da pedra, contornos
python3 baseApp.py imagem.jpg -d
```

**Quando usar o baseApp.py:**
- Ao ajustar `area_ponto` para um novo zoom ou câmera
- Para verificar se uma pedra específica está sendo reconhecida corretamente
- Para depurar erros de contagem sem interferência do vídeo ao vivo

---

## Configurações relevantes

- `area_ponto`: área mínima em pixels dos pontos do dominó — principal parâmetro
  caso esteja reconhecendo os contornos e retornando "0|0"
- Zoom e intervalo de leitura também são ajustáveis pela interface web em tempo real

## Estrutura

```
app.py          # Processamento de vídeo ao vivo e servidor Flask
baseApp.py      # Versão para imagens estáticas (desenvolvimento e calibração)
index.html      # Interface web do placar
```
