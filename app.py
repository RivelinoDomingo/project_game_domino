from flask import Flask, jsonify, render_template, request, Response
import cv2
import numpy as np
import math
import time # Precisamos importar o time
import threading # Para rodar a câmera em segundo plano

app = Flask(__name__)

# Configura a câmera (o Droidcam costuma criar um dispositivo no /dev/video0 ou similar)
# Se estiver usando Droidcam via WiFi, você pode até passar a URL aqui! Ex: cv2.VideoCapture("http://192.168.1.100:4747/video")
# camera = cv2.VideoCapture("http://192.168.1.135:5000/video")
camera = cv2.VideoCapture("http://192.168.1.135:5000/video?video_size=1920x1080")
# Tenta forçar a câmera a ler em HD (1280x720) ou Full HD (1920x1080)
# camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Variáveis globais para armazenar a última leitura da mesa
zoom_factor = 0.8
ultima_leitura_pedras = []
ultimo_tempo_processamento = 0
ultimo_frame_processado = None
INTERVALO_SEGUNDOS = 2.0 # Processa a mesa a cada 2 segundos
enviar_video = True      # Começa ligado
DISTANCIA_MINIMA = 30    # Distância minima entre as pedras
TEMPO_MEMORIA = 4.0
cache_pedras = []
modo_leitura = 'mesa'
tirar_foto_debug = False  # O nosso gatilho de foto debug
maos_jogadores = {
    'p1': [], 'p2': [], 'p3': [], 'p4': []
}
Zerou_mao = False
estado_intervalo = False
CP_INTERVALO_SEGUNDOS = 0

def ordenar_pontos(pts):
    # Inicializa uma lista de coordenadas que serão ordenadas
    # [top-left, top-right, bottom-right, bottom-left]
    rect = np.zeros((4, 2), dtype="float32")

    # O ponto superior-esquerdo terá a menor soma, o inferior-direito a maior
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # O ponto superior-direito terá a menor diferença, o inferior-esquerdo a maior
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def extrair_e_contar(img, rect_pedra):
    # Pega as 4 quinas da caixa verde
    box = cv2.boxPoints(rect_pedra)
    pts = ordenar_pontos(box)

    # Forçar a imagem a ficar sempre "em pé" (altura maior que largura)
    dist_0_1 = np.linalg.norm(pts[0] - pts[1]) # Largura superior
    dist_0_3 = np.linalg.norm(pts[0] - pts[3]) # Altura esquerda

    if dist_0_1 > dist_0_3:
        # Se estiver deitada, nós rotacionamos os pontos em 90 graus
        pts = np.array([pts[1], pts[2], pts[3], pts[0]], dtype="float32")

    # Criamos o "molde" perfeito de 40x80 pixels
    dst = np.array([
        [0, 0],
        [39, 0],
        [39, 79],
        [0, 79]
    ], dtype="float32")

    # Mágica: Estica a pedra inclinada da foto para caber no nosso molde perfeito
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (40, 80))

    # Guilhotina: Corta o molde ao meio
    metade_cima = warped[0:40, 0:40]
    metade_baixo = warped[40:80, 0:40]

    def contar_bolinhas(metade):
            gray = cv2.cvtColor(metade, cv2.COLOR_BGR2GRAY)

            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            # BLINDAGEM 1: Apagar as bordas da imagem (3 pixels)
            # Isso impede que a sombra da beirada do dominó seja contada como bolinha
            h, w = thresh.shape
            cv2.rectangle(thresh, (0, 0), (w, h), 0, 3)

            kernel = np.ones((2,2), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pontos = 0

            for c in contornos:
                area = cv2.contourArea(c)

                # A sua área super calibrada pelo GIMP! (Dei uma margem de segurança 25 a 85)
                if 30 < area < 85:
                    # BLINDAGEM 2: O filtro de formato (Circularidade)
                    perimetro = cv2.arcLength(c, True)
                    if perimetro == 0: continue

                    circularidade = 4 * np.pi * (area / (perimetro * perimetro))

                    # Se for redondo o suficiente (Círculo = 1.0, Quadrado ~0.78)
                    if circularidade > 0.4:
                        pontos += 1

            # BLINDAGEM 3: Trava matemática máxima de um dominó
            return min(pontos, 6)

    pts_cima = contar_bolinhas(metade_cima)
    pts_baixo = contar_bolinhas(metade_baixo)

    # Opcional: mostrar as pedras extraídas para você ver a mágica acontecendo (comente depois)
    # cv2.imshow("Pedra Extraida", warped)
    # cv2.waitKey(0)

    return pts_cima, pts_baixo

def validar_pedra_lisa(gray_img, rect_pedra):
    # 1. Obter os 4 cantos da caixa verde
    box = cv2.boxPoints(rect_pedra)
    box = np.int32(box)

    # Usa a sua função já existente para colocar os pontos na ordem certa
    pts = ordenar_pontos(box)

    # 2. "Esticar" a imagem para um retângulo perfeito (ex: 60x120 pixels)
    largura, altura = 60, 120
    pts_destino = np.array([
        [0, 0],
        [largura - 1, 0],
        [largura - 1, altura - 1],
        [0, altura - 1]
    ], dtype="float32")

    matriz = cv2.getPerspectiveTransform(pts, pts_destino)
    pedra_plana = cv2.warpPerspective(gray_img, matriz, (largura, altura))

    # 3. A Guilhotina: Divide exatamente no meio do traço
    metade_cima = pedra_plana[0:altura//2, 0:largura]
    metade_baixo = pedra_plana[altura//2:altura, 0:largura]

    # 4. Calcula o Brilho e a Textura de cada metade individualmente
    mean_c = cv2.meanStdDev(metade_cima)
    mean_b = cv2.meanStdDev(metade_baixo)

    brilho_c = mean_c[0][0]
    brilho_b = mean_b[0][0]

    # 5. A NOVA REGRA (Simetria de Brilho)
    diferenca_brilho = abs(brilho_c - brilho_b)

    # ========================================================
    # OS CRITÉRIOS DE APROVAÇÃO:
    # 1. As duas metades têm de ser claras (> 100)
    # 2. A diferença de luz entre elas tem de ser pequena (< 40)
    # ========================================================
    if (brilho_c > 100 and brilho_b > 100) and diferenca_brilho < 30:
        return True
    else:
        # Imprime no terminal o MOTIVO da rejeição para você poder calibrar!
        # print(f"👻 Fantasma rejeitado! Brilho (C:{brilho_c:.0f}, B:{brilho_b:.0f}) Dif: {diferenca_brilho:.0f}")
        # print(f"👻 Fantasma rejeitado! Brilho (C:{brilho_c:.0f}, B:{brilho_b:.0f}) | Dif: {diferenca_brilho:.0f} | Text(C:{textura_c:.0f}, B:{textura_b:.0f})")
        return False

# ====================================================================

def loop_da_camera():
    global ultima_leitura_pedras, ultimo_tempo_processamento, camera

    while True:
        sucesso, frame = camera.read()
        if not sucesso:
            print("Erro: Frame não processado.")
            print("Possível desconexão.")
            time.sleep(5)
            camera = cv2.VideoCapture("http://192.168.1.135:5000/video?video_size=1920x1080")
            continue

        tempo_atual = time.time()

        global modo_leitura, maos_jogadores, DISTANCIA_MINIMA, INTERVALO_SEGUNDOS, tirar_foto_debug
        global Zerou_mao, CP_INTERVALO_SEGUNDOS, estado_intervalo

        if not estado_intervalo:
            CP_INTERVALO_SEGUNDOS = INTERVALO_SEGUNDOS

        if modo_leitura != 'mesa':
            DISTANCIA_MINIMA = 10
            estado_intervalo = True
            INTERVALO_SEGUNDOS = 0.4   ## Maior velocidade na identificação da pedra dos jogadores
            if not Zerou_mao:
                maos_jogadores[modo_leitura] = None
                Zerou_mao = True
        else:
            DISTANCIA_MINIMA = 30
            INTERVALO_SEGUNDOS = CP_INTERVALO_SEGUNDOS
            estado_intervalo = False
            Zerou_mao = False

        # Só executa a Visão Computacional se já passou o nosso intervalo
        if tempo_atual - ultimo_tempo_processamento >= INTERVALO_SEGUNDOS:
            ultimo_tempo_processamento = tempo_atual

            img = frame.copy()
            if img is None:
                print("Erro: Imagem não encontrada.")
                return
            # --- APLICA O ZOOM DIGITAL ANTES DE TUDO ---
            if zoom_factor != 1.0:
                # Interpolação LINEAR mantém a qualidade ao dar zoom
                img = cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)

            if tirar_foto_debug:
                nome_arquivo = f"debug_mao_{modo_leitura}.jpeg"
                cv2.imwrite(nome_arquivo, img)
                print(f"📸 FOTO DE DEBUG SALVA NO PC: {nome_arquivo}")

                # Desarma o gatilho para não encher o seu HD de fotos iguais
                tirar_foto_debug = False

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_bh)

            _, mask_bh = cv2.threshold(blackhat, 100, 255, cv2.THRESH_BINARY)
            kernel_close = np.ones((2,2), np.uint8)
            mask_soldada = cv2.morphologyEx(mask_bh, cv2.MORPH_CLOSE, kernel_close)

            contours, _ = cv2.findContours(mask_soldada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            candidatos = []

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 20 or area > 300:
                    continue

                rect = cv2.minAreaRect(cnt)
                (cx, cy), (w_box, h_box), angle = rect

                if w_box == 0 or h_box == 0:
                    continue

                linha_comprimento = max(w_box, h_box)
                linha_espessura = min(w_box, h_box)
                ratio = linha_comprimento / linha_espessura

                # --- A NOVA BLINDAGEM DE TAMANHO ---
                # A sua pedra projetada tem 30px de largura.
                # O traço costuma ter entre 15px e 26px.
                # Vamos barrar qualquer coisa que seja maior que a própria pedra!
                LIMITE_MAX_TRACO = 32
                LIMITE_MIN_TRACO = 12 # Ignora cisco que o ratio achou que era linha
                LIMITE_EXPESSURA = 5

                # Adicionamos os limites de comprimento no IF principal
                if ratio > 2.0 and (LIMITE_MIN_TRACO <= linha_comprimento <= LIMITE_MAX_TRACO) and (1 <= linha_espessura <= LIMITE_EXPESSURA):

                    if w_box > h_box:
                        rect_pedra = ((cx, cy), (30, 61), angle)
                    else:
                        rect_pedra = ((cx, cy), (61, 30), angle)

                    # --- A SUA IDEIA APLICADA AQUI ---
                    # 1. Criar uma máscara preta do mesmo tamanho da imagem
                    mask_box = np.zeros(gray.shape, dtype=np.uint8)

                    # 2. Desenhar a nossa caixa verde preenchida de branco nessa máscara
                    box_pedra_pts = np.int32(cv2.boxPoints(rect_pedra))
                    cv2.fillPoly(mask_box, [box_pedra_pts], 255)

                    # 3. Calcular a média de brilho da imagem original SOMENTE DENTRO da caixa
                    # Retorna um valor de 0 (preto absoluto) a 255 (branco absoluto)
                    brilho_medio = cv2.mean(gray, mask=mask_box)[0]

                    # 4. Só aceitamos se a caixa for predominantemente clara (dominó)
                    # Você pode ajustar esse valor (80 a 110) dependendo da sua iluminação
                    if brilho_medio > 90:
                        candidatos.append({
                            'rect_pedra': rect_pedra,
                            'rect_traco': rect,
                            'centro': (cx, cy),
                            'brilho': brilho_medio  # Guardamos o brilho para o duelo final!
                        })

            # --- DUELO E FILTRO DE DISTÂNCIA ---
            # Ordenamos os candidatos do MAIS CLARO para o MAIS ESCURO.
            # Isso garante que caixas centralizadas perfeitas derrotem as caixas de borda.
            candidatos.sort(key=lambda x: x['brilho'], reverse=True)

            # ==========================================
            # PASSO 1: Remover Duplicatas (Caixas na mesma pedra)
            # ==========================================
            pedras_unicas = []
            for cand in candidatos:
                cx1, cy1 = cand['centro']
                duplicata = False

                for p in pedras_unicas:
                    cx2, cy2 = p['centro']
                    if math.hypot(cx2 - cx1, cy2 - cy1) < DISTANCIA_MINIMA:
                        duplicata = True
                        break

                if not duplicata:
                    pedras_unicas.append(cand)

            # ==========================================
            # PASSO 2: O Filtro da "Área de Influência" (O Maior Bando)
            # ==========================================
            DISTANCIA_CONEXAO = 100  # Tamanho da "Área de influência" de cada pedra

            visitados = set()
            todos_os_bandos = []

            for i, p1 in enumerate(pedras_unicas):
                # Se essa pedra já entrou num bando antes, ignoramos
                if i in visitados:
                    continue

                # Começamos um novo bando com essa pedra
                bando_atual = [p1]
                visitados.add(i)

                # A "Fila de Expansão" (vai checar os amigos dos amigos)
                fila_de_expansao = [p1]

                while fila_de_expansao:
                    pedra_foco = fila_de_expansao.pop(0)
                    cx_foco, cy_foco = pedra_foco['centro']

                    # Procura novos amigos para puxar para o bando
                    for j, p2 in enumerate(pedras_unicas):
                        if j not in visitados:
                            cx2, cy2 = p2['centro']
                            dist = math.hypot(cx2 - cx_foco, cy2 - cy_foco)

                            # Se a pedra está dentro da área de influência, entra pro bando!
                            if dist <= DISTANCIA_CONEXAO:
                                bando_atual.append(p2)
                                visitados.add(j)
                                # Coloca ela na fila para a área de influência dela também ser checada!
                                fila_de_expansao.append(p2)

                # Guarda o bando que acabamos de formar
                todos_os_bandos.append(bando_atual)

            # ==========================================
            # PASSO 3: Sobrevivência do Mais Forte
            # ==========================================
            if todos_os_bandos:
                # A função max() com 'key=len' pega automaticamente a lista que tem mais itens!
                maior_bando = max(todos_os_bandos, key=len)
                pedras_aprovadas = maior_bando
            else:
                pedras_aprovadas = []

            print(f"Pedras únicas encontradas: {len(pedras_aprovadas)}")

            # Ordena as pedras de cima para baixo (pelo eixo Y do centro)
            pedras_aprovadas.sort(key=lambda x: x['centro'][1])

            # =================================================================
            # --- MEMÓRIA INDIVIDUAL (O MAPA DA MESA) ---
            # =================================================================

            lista_final = []
            out = img.copy()

            global ultima_leitura_pedras, ultimo_frame_processado
            global cache_pedras, enviar_video
            DISTANCIA_TOLERANCIA = 5 # Se a pedra está no mesmo lugar (margem de 30px), é a mesma.

            pedras_vistas_agora = []
            valor_pedra = None

            for d in pedras_aprovadas:
                # Tratando de forma direta quando no modo de seleção da maõ do jogador
                if modo_leitura != 'mesa':
                    # Pega a contagem real
                    pts_cima, pts_baixo = extrair_e_contar(img, d['rect_pedra'])
                    # Adiciona na lista que vai para a Web
                    soma_pts = pts_cima + pts_baixo
                    if soma_pts <= 2:
                        if not validar_pedra_lisa(gray, d['rect_pedra']):
                            # Se for falso positivo (ex: linha na mesa), pula pro próximo laço!
                            # print(f"Pedra Rejeitada: {texto}")
                            continue
                    valor_pedra = f"{pts_cima}|{pts_baixo}"

                    if enviar_video:
                        # --- DESENHA AS CAIXAS PARA A WEB VER ---
                        box_traco = np.int32(cv2.boxPoints(d['rect_traco']))
                        cv2.drawContours(out, [box_traco], 0, (255, 0, 0), 2)
                        box_pedra = np.int32(cv2.boxPoints(d['rect_pedra']))
                        cv2.drawContours(out, [box_pedra], 0, (0, 255, 0), 2)

                        # Opcional: Escreve o número da pedra lida na própria imagem
                        cx, cy = int(d['centro'][0]), int(d['centro'][1])
                        cv2.putText(out, f"{pts_cima}|{pts_baixo}", (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:

                    cx_nova, cy_nova = d['centro']
                    pedra_reconhecida_memoria = None
                    menor_distancia = float('inf')

                    # 1. Procura qual pedra da memória está mais perto desta caixa atual
                    for p_mem in cache_pedras:
                        cx_mem, cy_mem = p_mem['centro']
                        dist = math.hypot(cx_mem - cx_nova, cy_mem - cy_nova)

                        if dist < menor_distancia:
                            menor_distancia = dist
                            if dist < DISTANCIA_TOLERANCIA:
                                pedra_reconhecida_memoria = p_mem

                    # 2. Decide se aproveita a memória ou se gasta CPU para recalcular
                    if pedra_reconhecida_memoria:
                        # ACHOU NO MAPA: É a mesma pedra de antes! Copia o valor.
                        valor_pedra = pedra_reconhecida_memoria['valor']
                    else:
                        # LUGAR NOVO: Pedra recém-colocada (ou a mesa foi arrastada). Recalcula!
                        pts_cima, pts_baixo = extrair_e_contar(img, d['rect_pedra'])
                        soma_pts = pts_cima + pts_baixo
                        if soma_pts <= 2:
                            if not validar_pedra_lisa(gray, d['rect_pedra']):
                                # Se for falso positivo (ex: linha na mesa), pula pro próximo laço!
                                # print(f"Pedra Rejeitada: {texto}")
                                continue
                        valor_pedra = f"{pts_cima}|{pts_baixo}"

                # 3. Adiciona na lista de hoje (atualizando a coordenada exata para não haver "drift")

                pedras_vistas_agora.append({
                    'centro': (cx_nova, cy_nova),
                    'rect_pedra': d['rect_pedra'],
                    'rect_traco': d['rect_traco'],
                    'valor': valor_pedra,
                    'ultima_vez_vista': tempo_atual
                })

            # 4. RECUPERAÇÃO DE FANTASMAS (Mão na frente da câmera)
            # Se uma pedra antiga não foi vista agora, mas tem tempo de vida, nós a mantemos viva
            if modo_leitura == 'mesa':
                for p_mem in cache_pedras:
                    ja_vista = False
                    for p_agora in pedras_vistas_agora:
                        # Verifica se o espaço dela já foi ocupado
                        dist = math.hypot(p_mem['centro'][0] - p_agora['centro'][0], p_mem['centro'][1] - p_agora['centro'][1])
                        if dist < DISTANCIA_TOLERANCIA:
                            ja_vista = True
                            break

                    if not ja_vista and (tempo_atual - p_mem['ultima_vez_vista']) <= TEMPO_MEMORIA:
                        # Se ninguém tomou o lugar dela e ela ainda tem tempo, injeta ela de volta!
                        pedras_vistas_agora.append(p_mem)

            # Atualiza o mapa oficial
            cache_pedras = pedras_vistas_agora
            # =================================================================

            lista_final = [p['valor'] for p in cache_pedras]

            ultima_leitura_pedras = lista_final

            # ONDE ESTAMOS A OLHAR?
            if modo_leitura == 'mesa':
                ultima_leitura_pedras = lista_final
            else:
                # Se estivermos a escanear um jogador, salvamos as pedras na mão dele!
                # Só atualizamos se a leitura estiver estável (para evitar salvar ruído de movimento)
                maos_jogadores[modo_leitura] = lista_final

            # --- OTIMIZAÇÃO: Só gasta CPU com desenhos e JPG se a página pedir! ---
            if enviar_video:
                for p in cache_pedras:
                    box_traco = np.int32(cv2.boxPoints(p['rect_traco']))
                    cv2.drawContours(out, [box_traco], 0, (255, 0, 0), 2)
                    box_pedra = np.int32(cv2.boxPoints(p['rect_pedra']))
                    cv2.drawContours(out, [box_pedra], 0, (0, 255, 0), 2)
                    cx, cy = int(p['centro'][0]), int(p['centro'][1])
                    cv2.putText(out, p['valor'], (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                sucesso_encode, buffer = cv2.imencode('.jpg', out)
                if sucesso_encode:
                    ultimo_frame_processado = buffer.tobytes()
            else:
                ultimo_frame_processado = None

# ====================================================================
# ROTAS DA WEB (A API)
# ====================================================================

def gerar_frames():
    global ultimo_frame_processado
    while True:
        if ultimo_frame_processado is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + ultimo_frame_processado + b'\r\n')

            # ATENÇÃO AQUI: Forçar o streaming a rodar a ~10 FPS
            # Sem isso, ele tenta mandar frames na velocidade da luz e trava o PC!
            time.sleep(0.1)
        else:
            # Se ainda não houver foto, espera 0.1s e tenta de novo
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    # Essa rota devolve o vídeo ao vivo!
    return Response(gerar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/toggle_video', methods=['POST'])
def toggle_video():
    global enviar_video
    dados = request.get_json()
    enviar_video = dados.get('ativar', True)
    print(f"Transmissão de vídeo: {'LIGADA' if enviar_video else 'DESLIGADA'}")
    return jsonify({"status": "sucesso"})

@app.route('/api/config', methods=['POST'])
def atualizar_config():
    global TEMPO_MEMORIA, INTERVALO_SEGUNDOS
    dados = request.get_json()

    # Atualiza as variáveis globais em tempo real!
    TEMPO_MEMORIA = float(dados.get('tempo_memoria', TEMPO_MEMORIA))
    INTERVALO_SEGUNDOS = float(dados.get('intervalo_segundos', INTERVALO_SEGUNDOS))

    print(f"⚙️ Config atualizada: Memória = {TEMPO_MEMORIA}s | Intervalo = {INTERVALO_SEGUNDOS}s")
    return jsonify({"status": "sucesso"})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/zoom', methods=['POST'])
def atualizar_zoom():
    global zoom_factor
    # Recebe o valor do slider enviado pelo Javascript
    dados = request.get_json()
    zoom_factor = float(dados.get('zoom', 1.0))
    print(f"Zoom atualizado para: {zoom_factor}x")
    return jsonify({"status": "sucesso"})

@app.route('/api/mesa')
def api_mesa():
    # O Flask só pega a última leitura salva! Retorna em milissegundos.
    return jsonify({
        "status": "sucesso",
        "quantidade": len(ultima_leitura_pedras),
        "pedras": ultima_leitura_pedras
    })

@app.route('/api/set_modo', methods=['POST'])
def set_modo():
    global modo_leitura, tirar_foto_debug
    dados = request.get_json()
    modo_leitura = dados.get('modo', 'mesa')

    # Se fomos ler a mão de alguém, armamos o gatilho da foto!
    if modo_leitura != 'mesa':
        tirar_foto_debug = True

    print(f"📷 Câmera redirecionada para ler: {modo_leitura.upper()}")
    return jsonify({"status": "sucesso", "modo": modo_leitura})

@app.route('/api/estado_jogo')
def estado_jogo():
    # Esta rota envia TUDO (mesa e jogadores) para o HTML desenhar de uma vez só
    return jsonify({
        "modo_atual": modo_leitura,
        "mesa": ultima_leitura_pedras,
        "jogadores": maos_jogadores
    })

if __name__ == '__main__':
    # Inicia o loop da câmera em uma thread separada para não travar o servidor Web!
    t = threading.Thread(target=loop_da_camera, daemon=True)
    t.start()

    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    # use_reloader=False é vital quando se usa câmera com Flask, senão ele tenta ligar a câmera duas vezes e trava.
