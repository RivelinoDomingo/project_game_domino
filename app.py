from flask import Flask, jsonify, render_template, request, Response
import cv2
import numpy as np
import math
import time
import threading
import atexit
from collections import deque

app = Flask(__name__)

# Configurações otimizadas
device = 'http://192.168.1.135:5000/video?video_size=1920x1080'
zoom_factor = 1.0
ultima_leitura_pedras = []
ultimo_tempo_processamento = 0
ultimo_frame_processado = None
INTERVALO_SEGUNDOS = 2.0
executando_servidor = True
enviar_video = False
DISTANCIA_MINIMA = 37
TEMPO_MEMORIA = 4.0
cache_pedras = []
modo_leitura = 'mesa'
actions = {'action': 'none', 'action1': 'none', 'action2': 'none'}
resetMaoPlayers = False
tirar_foto_debug = False
maos_jogadores = {'p1': [], 'p2': [], 'p3': [], 'p4': []}
Zerou_mao = False
estado_intervalo = False
CP_INTERVALO_SEGUNDOS = 0
duplicada = None
zoom_reset = False
zoom_change = False
modoAuto = False

# Cache para frames para evitar processamento repetido
frame_buffer = deque(maxlen=2)
ultimo_frame_valido = None
falhas_consecutivas = 0
MAX_FALHAS = 10


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

    brilho_c = mean_c[0][0].item()
    brilho_b = mean_b[0][0].item()

    # 5. A NOVA REGRA (Simetria de Brilho)
    diferenca_brilho = abs(brilho_c - brilho_b)

    # ========================================================
    # OS CRITÉRIOS DE APROVAÇÃO:
    # 1. As duas metades têm de ser claras (> 100)
    # 2. A diferença de luz entre elas tem de ser pequena (< 40)
    # ========================================================
    if (brilho_c > 150 and brilho_b > 150) and diferenca_brilho < 30:
        return True
    else:
        # Imprime no terminal o MOTIVO da rejeição para você poder calibrar!
        print(f"👻 Fantasma rejeitado! Brilho (C:{brilho_c:.0f}, B:{brilho_b:.0f}) Dif: {diferenca_brilho:.0f}")
        return False

def valor_ja_existe(valor_procurado, modo_atual, pedras_ja_vistas_neste_frame):
    global maos_jogadores

    # 1. Cria a versão invertida da pedra
    partes = valor_procurado.split('|')
    valor_invertido = f"{partes[1]}|{partes[0]}"

    # 2. EVITA DUPLICATAS NO MESMO FRAME
    for p in pedras_ja_vistas_neste_frame:
        if p['valor'] == valor_procurado or p['valor'] == valor_invertido:
            return True

    # 3. SE ESTIVERMOS LENDO A MESA: Nunca bloqueia!
    if modo_atual == 'mesa':
        return False

    # 4. VERIFICA AS MÃOS DOS OUTROS JOGADORES
    for player, pedras_da_mao in maos_jogadores.items():
        # Pula o próprio jogador que estamos lendo
        if player == modo_atual:
            continue

        # Verifica se a mão existe e é uma lista
        if pedras_da_mao is None or not isinstance(pedras_da_mao, list):
            continue

        # 🔧 CORREÇÃO AQUI: Percorre a lista de dicionários
        for pedra_dict in pedras_da_mao:
            valor_na_mao = pedra_dict.get('valor', '')
            if valor_na_mao == valor_procurado or valor_na_mao == valor_invertido:
                # print(f"🚫 Bloqueado: {valor_procurado} já está na mão de {player} como {valor_na_mao}")
                return True

    return False

def processar_grupos(numeros, percentual_max=10, min_por_grupo=5, max_por_grupo=10):
    """
    Agrupa números próximos (dentro do percentual) e retorna as médias dos grupos válidos

    Args:
        numeros: lista de números
        percentual_max: diferença máxima permitida em % (padrão: 10)
        min_por_grupo: tamanho mínimo do grupo (padrão: 5)
        max_por_grupo: tamanho máximo do grupo (padrão: 10)

    Returns:
        Lista com as médias dos grupos que atendem aos critérios
    """
    if not numeros:
        return []

    # Ordena os números
    numeros_ordenados = sorted(numeros)
    grupos = []
    grupo_atual = [numeros_ordenados[0]]

    # Primeira etapa: formar grupos baseado na diferença percentual
    for i in range(1, len(numeros_ordenados)):
        # Calcula diferença percentual em relação ao PRIMEIRO elemento do grupo
        primeiro_do_grupo = grupo_atual[0]
        diferenca_percentual = abs(numeros_ordenados[i] - primeiro_do_grupo) / primeiro_do_grupo * 100

        if diferenca_percentual <= percentual_max and len(grupo_atual) < max_por_grupo:
            grupo_atual.append(numeros_ordenados[i])
        else:
            # Salva o grupo atual se atender ao tamanho mínimo
            if len(grupo_atual) >= min_por_grupo:
                grupos.append(grupo_atual)
            grupo_atual = [numeros_ordenados[i]]

    # Adiciona o último grupo
    if len(grupo_atual) >= min_por_grupo:
        grupos.append(grupo_atual)

    # Calcula as médias dos grupos
    medias = [sum(grupo) / len(grupo) for grupo in grupos]

    return medias

def inicializar_camera():
    """Inicializa a câmera com tentativas e timeout"""
    global camera
    try:
        camera = cv2.VideoCapture(device)
        if not camera.isOpened():
            print("⚠️ Falha ao abrir câmera, tentando novamente...")
            time.sleep(1)
            camera = cv2.VideoCapture(device)

        # Configurações para reduzir buffer e latência
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return camera.isOpened()
    except Exception as e:
        print(f"❌ Erro ao inicializar câmera: {e}")
        return False

# Inicializa câmera
camera = None
inicializar_camera()

def ler_frame_com_timeout(timeout=1):
    """Lê frame com timeout para não travar"""
    global camera, falhas_consecutivas, ultimo_frame_valido

    if camera is None or not camera.isOpened():
        if not inicializar_camera():
            return None

    try:
        # Tenta ler com timeout (usando thread)
        sucesso = False
        frame = None

        def ler():
            nonlocal sucesso, frame
            try:
                sucesso, frame = camera.read()
            except:
                sucesso = False
                frame = None

        thread = threading.Thread(target=ler)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            print("⚠️ Timeout na leitura da câmera")
            return ultimo_frame_valido  # Retorna último frame válido

        if sucesso and frame is not None:
            falhas_consecutivas = 0
            ultimo_frame_valido = frame
            return frame
        else:
            falhas_consecutivas += 1
            if falhas_consecutivas >= MAX_FALHAS:
                print(f"❌ {MAX_FALHAS} falhas consecutivas, reiniciando câmera...")
                if camera:
                    camera.release()
                time.sleep(1)
                inicializar_camera()
                falhas_consecutivas = 0
            return ultimo_frame_valido

    except Exception as e:
        print(f"Erro na leitura: {e}")
        return ultimo_frame_valido

def loop_da_camera():
    global ultimo_tempo_processamento, executando_servidor
    global ultimo_frame_processado

    tempo_ultimo_frame = time.time()
    frames_sem_processar = 0

    while executando_servidor:
        try:
            # Lê frame com timeout
            frame = ler_frame_com_timeout(0.5)

            if frame is None:
                time.sleep(0.1)
                continue

            tempo_atual = time.time()
            frames_sem_processar += 1

            # Processa apenas no intervalo configurado
            if tempo_atual - ultimo_tempo_processamento >= INTERVALO_SEGUNDOS:
                ultimo_tempo_processamento = tempo_atual
                frames_sem_processar = 0

                # Processa o frame
                processar_frame(frame, tempo_atual)

            # Só envia vídeo se necessário
            if enviar_video and ultimo_frame_processado is not None:
                # Limita FPS do stream para não sobrecarregar
                if time.time() - tempo_ultimo_frame > 0.1:
                    tempo_ultimo_frame = time.time()
            else:
                # Pequena pausa para não consumir CPU
                time.sleep(0.05)

        except Exception as e:
            print(f"Erro no loop principal: {e}")
            time.sleep(0.5)

def processar_frame(img, tempo_atual):
    """Processa o frame de forma otimizada"""
    global ultima_leitura_pedras, ultimo_frame_processado, duplicada
    global cache_pedras, resetMaoPlayers, maos_jogadores
    global modo_leitura, tirar_foto_debug, enviar_video, zoom_factor

    # Rotaciona a imagem para ficar mais adequando à mesa
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    # Aplica zoom se necessário
    if zoom_factor != 1.0:
        img = cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor,
                        interpolation=cv2.INTER_LINEAR)

    # Reset das mãos se necessário
    if resetMaoPlayers:
        print("Resetando mãos dos jogadores...")
        resetMaoPlayers = False
        for player in maos_jogadores:
            maos_jogadores[player] = []

    # Configura intervalos dinâmicos
    global DISTANCIA_MINIMA, INTERVALO_SEGUNDOS, CP_INTERVALO_SEGUNDOS
    global estado_intervalo, Zerou_mao

    if modo_leitura != 'mesa':
        DISTANCIA_MINIMA = 25
        if not estado_intervalo:
            CP_INTERVALO_SEGUNDOS = INTERVALO_SEGUNDOS
            INTERVALO_SEGUNDOS = 0.4
            estado_intervalo = True
        if not Zerou_mao:
            maos_jogadores[modo_leitura] = []
            Zerou_mao = True
    else:
        DISTANCIA_MINIMA = 37
        if estado_intervalo:
            INTERVALO_SEGUNDOS = CP_INTERVALO_SEGUNDOS
            estado_intervalo = False
        Zerou_mao = False

    # Foto debug
    if tirar_foto_debug:
        cv2.imwrite(f"debug_mao_{modo_leitura}.jpeg", img)
        print(f"📸 Foto salva: debug_mao_{modo_leitura}.jpeg")
        tirar_foto_debug = False

    # Processamento de visão computacional
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Usa kernel menor para processamento mais rápido
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_bh)

    _, mask_bh = cv2.threshold(blackhat, 100, 255, cv2.THRESH_BINARY)
    kernel_close = np.ones((2,2), np.uint8)
    mask_soldada = cv2.morphologyEx(mask_bh, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(mask_soldada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # Processa contornos (restante do código similar ao original, mas otimizado)
    candidatos = []
    metricas = []
    mediaFinal = None


    global zoom_reset, modoAuto, processar_grupos, zoom_change

    if modoAuto and not zoom_reset:
        zoom_reset = True
        zoom_factor = 1.0

    if zoom_reset:
        zoom_reset = False

    # Limita o número de contornos processados
    if modoAuto:
        max_contours = 200
        max_area = 800
    else:
        max_contours = 300
        max_area = 100

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10 or area > max_area:
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w_box, h_box), angle = rect

        if w_box == 0 or h_box == 0:
            continue

        linha_comprimento = max(w_box, h_box)
        linha_espessura = min(w_box, h_box)
        ratio = linha_comprimento / linha_espessura

        if modoAuto:
            if 15.0 > ratio > 5.0:
                metricas.append(linha_comprimento)
                # print(f"Comprimento: {linha_comprimento}    -    Espessura: {linha_espessura}    -    Ratio: {ratio}   ##  Zoom: {zoom_factor}")
                # time.sleep(0.5)

        if ratio > 6.0 and not modoAuto and 18 <= linha_comprimento <= 28 and 1 <= linha_espessura <= 6:
            if w_box > h_box:
                rect_pedra = ((cx, cy), (30, 61), angle)
            else:
                rect_pedra = ((cx, cy), (61, 30), angle)

            # Verifica brilho
            mask_box = np.zeros(gray.shape, dtype=np.uint8)
            box_pedra_pts = np.int32(cv2.boxPoints(rect_pedra))
            cv2.fillPoly(mask_box, [box_pedra_pts], 255)
            brilho_medio = cv2.mean(gray, mask=mask_box)[0]

            if brilho_medio > 90:
                candidatos.append({
                    'rect_pedra': rect_pedra,
                    'rect_traco': rect,
                    'centro': (cx, cy),
                    'brilho': brilho_medio
                })


    # Auto calibração
    mediaFinal = 0
    medias = []
    if modoAuto:
        medias = processar_grupos(metricas)
        print("Iniciando a calibração!!!")

        for i in medias:
            if i >= 7:
                # print(f"Valor médio de: {i}px")
                mediaFinal = i

    TAMANHO_IDEAL = 24.0 # O tamanho que o algoritmo ama ler

    if modoAuto and len(medias) > 0:
        # Pega a média mais relevante (geralmente a maior ou a do grupo com mais itens)
        # Vamos supor que a sua função retorna a lista de médias válidas
        media_tracos = mediaFinal

        fator_correcao = TAMANHO_IDEAL / media_tracos

        # global zoom_factor
        zoom_factor = zoom_factor * fator_correcao

        print(f"🎯 Auto-Calibragem Concluída!")
        print(f"Traço atual: {media_tracos:.1f} -> Alvo: {TAMANHO_IDEAL}")
        print(f"Novo Zoom ajustado para: {zoom_factor:.2f}x")

        # Desliga o modo auto para o jogo normal continuar com o zoom perfeito!
        actions['action1'] = None
        zoom_change = True
        modoAuto = False

    # Remove duplicatas (continuação do código original)
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

    print(f"Pedras encontradas: {len(pedras_aprovadas)}")

    # Ordena as pedras de cima para baixo (pelo eixo Y do centro)
    pedras_aprovadas.sort(key=lambda x: x['centro'][1])

    # =================================================================
    # --- MEMÓRIA INDIVIDUAL (O MAPA DA MESA) ---
    # =================================================================

    lista_final = []
    out = img.copy()

    global ultima_leitura_pedras, ultimo_frame_processado, duplicada
    global cache_pedras, enviar_video, valor_ja_existe

    DISTANCIA_TOLERANCIA = 5 # Se a pedra está no mesmo lugar (margem de 5px), é a mesma.

    pedras_vistas_agora = []
    valor_pedra = None
    duplicada = None

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
                    pedras_aprovadas.remove(d)
                    continue
            valor_pedra = f"{pts_cima}|{pts_baixo}"
            cx_nova, cy_nova = d['centro']

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
                        pedras_aprovadas.remove(d)
                        # Se for falso positivo (ex: linha na mesa), pula pro próximo laço!
                        # print(f"Pedra Rejeitada: {texto}")
                        continue
                valor_pedra = f"{pts_cima}|{pts_baixo}"
        # 3. Adiciona na lista de hoje (atualizando a coordenada exata para não haver "drift")
        if not valor_ja_existe(valor_pedra, modo_leitura, pedras_vistas_agora):
            pedras_vistas_agora.append({
                'centro': (cx_nova, cy_nova),
                'rect_pedra': d['rect_pedra'],
                'rect_traco': d['rect_traco'],
                'valor': valor_pedra,
                'ultima_vez_vista': tempo_atual
            })
            # duplicada = None
        else:
            # Imprime apenas se NÃO for a mesa para evitar poluir o log
            if modo_leitura != 'mesa':
                print(f"🚫 Duplicata rejeitada: {valor_pedra} (já pertence a outro ou lida duas vezes)")
                duplicada = valor_pedra
                continue

    print(f"Pedras aprovadas: {len(pedras_aprovadas)}")

    ## Bloco de debug de dados das pedras
    # global ler_info
    # if ler_info and modo_leitura != 'mesa':
    #     print(pedras_vistas_agora)
    #     ler_info = False

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

    for p in cache_pedras:
        cx, cy = p['rect_pedra'][0]

        # Desempacotamos a Largura (w) e a Altura (h)
        w, h = p['rect_pedra'][1]

        angulo_cv = p['rect_pedra'][2]

        # --- A INTELIGÊNCIA DE ORIENTAÇÃO ---
        # O CSS desenha as pedras "em pé" por padrão (0 graus).
        if w < h:
            # Pedra EM PÉ. O OpenCV deu ~90 graus.
            # Subtraímos 90 para o CSS entender que é 0 graus (Vertical)
            angulo_corrigido = angulo_cv - 90
        else:
            # Pedra DEITADA (ex: o 6|6). A largura é maior que a altura.
            # O OpenCV também deu ~90 graus, mas nós MANTEMOS os 90
            # para o CSS girar a pedra e deitá-la na tela!
            angulo_corrigido = angulo_cv

        lista_final.append({
            'valor': p['valor'],
            'x': cx,
            'y': cy,
            'angulo': angulo_corrigido
        })

    ultima_leitura_pedras = lista_final

    # ONDE ESTAMOS A OLHAR?
    if modo_leitura == 'mesa':
        ultima_leitura_pedras = lista_final
    else:
        # Se estivermos a escanear um jogador, salvamos as pedras na mão dele!
        # Só atualizamos se a leitura estiver estável (para evitar salvar ruído de movimento)
        if len(lista_final) == 7:
            maos_jogadores[modo_leitura] = lista_final
            print(f"✅ Mão de {modo_leitura} atualizada com {len(lista_final)} pedras")






    # Prepara frame para streaming
    if enviar_video:
        out = img.copy()
        for p in cache_pedras[:20]:  # Limita desenho
            try:
                box_traco = np.int32(cv2.boxPoints(p['rect_traco']))
                cv2.drawContours(out, [box_traco], 0, (255, 0, 0), 1)
                box_pedra = np.int32(cv2.boxPoints(p['rect_pedra']))
                cv2.drawContours(out, [box_pedra], 0, (0, 255, 0), 1)
            except:
                pass

        sucesso_encode, buffer = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if sucesso_encode:
            ultimo_frame_processado = buffer.tobytes()
    else:
        ultimo_frame_processado = None

    print(f"Pedras processadas: {len(lista_final)}")

# ====================================================================
# ENCERRAMENTO SEGURO
# ====================================================================

def liberar_recursos():
    global executando_servidor
    print("\n🛑 Recebido sinal de parada! Avisando a câmera...")

    # 1. Avisa a thread da câmera para parar o loop
    executando_servidor = False

    # 2. Espera meio segundo para a thread ter tempo de fechar o OpenCV
    time.sleep(0.5)

    print("🛑 Servidor encerrado.")

# Registra a função para rodar automaticamente quando o app for fechado
atexit.register(liberar_recursos)

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

@app.route('/api/action_exec', methods=['POST'])
def action_exec():
    global actions, resetMaoPlayers, modoAuto, zoom_factor
    dados = request.get_json()
    actions['action'] = dados.get('action')
    actions['action1'] = dados.get('action1')
    actions['action2'] = dados.get('action2')

    # Se fomos ler a mão de alguém, armamos o gatilho da foto!
    if actions['action'] == 'reset':
        resetMaoPlayers = True
        return jsonify({"status": "sucesso"})

    if actions['action1'] == 'calibrar':
        modoAuto = True
        return jsonify({"status": "sucesso"})

    if actions['action2'] == 'valor_zoom':
        return jsonify({"status": "sucesso", "zoom": zoom_factor})



@app.route('/api/estado_jogo')
def estado_jogo():
    global duplicada, zoom_change
    # Esta rota envia TUDO (mesa e jogadores) para o HTML desenhar de uma vez só
    zoom_state = None
    if zoom_change:
        zoom_state = 'zoom_change'
        zoom_change = False
    return jsonify({
        "modo_atual": modo_leitura,
        "mesa": ultima_leitura_pedras,
        "jogadores": maos_jogadores,
        "duplicada": duplicada,
        "zoom_state": zoom_state
    })

if __name__ == '__main__':
    # Inicia o loop da câmera em uma thread separada para não travar o servidor Web!
    t = threading.Thread(target=loop_da_camera, daemon=True)
    t.start()

    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    # use_reloader=False é vital quando se usa câmera com Flask, senão ele tenta ligar a câmera duas vezes e trava.
