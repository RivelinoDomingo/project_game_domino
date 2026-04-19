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
# device = 'http://192.168.1.100:5000/video?video_size=1920x1080'
device = '/home/rivelino/Downloads/rec_2026-04-07_21-49.mp4'
# device = '/home/rivelino/Git/project_game_domino/teste_colocamento_de_pedras.mp4'
zoom_factor = 1.2
ultima_leitura_pedras = []
ultimo_tempo_processamento = 0
ultimo_frame_processado = None
INTERVALO_SEGUNDOS = 0.5
executando_servidor = True
enviar_video = True
DISTANCIA_MINIMA = 37
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

debug_mode = False

# Cache para frames para evitar processamento repetido
frame_buffer = deque(maxlen=2)
ultimo_frame_valido = None
falhas_consecutivas = 0
MAX_FALHAS = 10
conf_busca = False
cord_cont = (0, 0)
area_base = 0

CONFIG_VALES = {
    'distancia_filtro': 15,
    'distancia_mov': 15,
    'distancia_corte': 62,
    'tamanho_kernel_morfologia': 15, # Novo parâmetro para o tamanho da fenda a ser fechada
    'area_max': 2000,                # Area maxima das pedras
    'area_min': 800,
    'area_ponto': 30,
    'distancia_conexao': 120,
}


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
    box = cv2.boxPoints(rect_pedra)
    pts = ordenar_pontos(box)

    dist_0_1 = np.linalg.norm(pts[0] - pts[1])
    dist_0_3 = np.linalg.norm(pts[0] - pts[3])

    if dist_0_1 > dist_0_3:
        pts = np.array([pts[1], pts[2], pts[3], pts[0]], dtype="float32")

    dst = np.array([[0, 0], [39, 0], [39, 79], [0, 79]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (40, 80))

    contornos, _ = cv2.findContours(warped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    zero_local = False
    if contornos:
        area = 0.0
        for c in contornos:
            area += cv2.contourArea(c)
            if area >= 41:
                zero_local = True
                break


    metade_cima = warped[0:40, 0:40]
    metade_baixo = warped[40:80, 0:40]

    def contar_bolinhas(metade):
        # gray = cv2.cvtColor(metade, cv2.COLOR_BGR2GRAY)
        # thresh = cv2.adaptiveThreshold(metade, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        _, thresh = cv2.threshold(metade, 250, 255, cv2.THRESH_BINARY)

        # h, w = thresh.shape
        # cv2.rectangle(thresh, (0, 0), (w, h), 0, 3)

        # kernel = np.ones((2, 2), np.uint8)
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # cv2.imshow("0 -- Meio pedra", thresh)
        # cv2.waitKey()

        contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # zero_local = False
        # if contornos:
        #     maior_cnt = cv2.contourArea(max(contornos, key=cv2.contourArea))
        #     # CORREÇÃO: Usando o zoom ao quadrado
        #     if maior_cnt >= 15:
        #         zero_local = True

        pontos = 0
        # CORREÇÃO: Usando o zoom ao quadrado
        point_area = CONFIG_VALES['area_ponto']

        for c in contornos:
            area = cv2.contourArea(c)
            circularidade = 0.0
            if point_area * 0.6 < area < point_area * 2.1:
                perimetro = cv2.arcLength(c, True)
                if perimetro == 0:
                    continue
                circularidade = 4 * np.pi * (area / (perimetro * perimetro))
                if circularidade >= 0.5:
                    pontos += 1
            # print(f"Valor de Área={area}, Circularidade={circularidade}")

        return min(pontos, 6)

    pts_cima = contar_bolinhas(metade_cima)
    pts_baixo = contar_bolinhas(metade_baixo)

    return pts_cima, pts_baixo, zero_local

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
        # camera = cv2.VideoCapture(device, cv2.CAP_FFMPEG)
        camera = cv2.VideoCapture(device)
        if not camera.isOpened():
            print("⚠️ Falha ao abrir câmera, tentando novamente...")
            time.sleep(1)
            # camera = cv2.VideoCapture(device, cv2.CAP_FFMPEG)
            camera = cv2.VideoCapture(device)


        # Configurações para reduzir buffer e latência
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return camera.isOpened()
    except Exception as e:
        print(f"❌ Erro ao inicializar câmera: {e}")
        return False

def nova_pedra(mask_filtrada, area_max_2, cord):
    # print("entrou na função nova_pedra")

    contours_1, _ = cv2.findContours(mask_filtrada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # PROTEÇÃO: Se não achou nenhum contorno na tela, aborta a função sem quebrar
    if not contours_1:
        return False, 0, (0, 0)

    # CORREÇÃO: Pega o maior contorno baseado na Área, não no array
    maior_cnt = max(contours_1, key=cv2.contourArea)
    area_max_1 = cv2.contourArea(maior_cnt)

    # CORREÇÃO: Limite de área já com zoom fatorado (conforme discutimos na outra interação)
    fator_area = zoom_factor ** 2
    area_test = CONFIG_VALES['area_min'] * fator_area

    # minAreaRect retorna (centro(x,y), tamanho(w,h), angulo)
    new_cord, _, _ = cv2.minAreaRect(maior_cnt)

    mov_limite = CONFIG_VALES['distancia_mov'] * zoom_factor

    if not area_max_1 or not area_max_2:
        return False, 0, (0, 0) # CORREÇÃO: removido o tuple(0, 0)

    # Checa alteração brusca de área (Pedra entrou ou saiu da mesa)
    if area_max_1 > (area_max_2 + area_test) or area_max_1 < (area_max_2 - area_test):
        return True, area_max_1, new_cord

    # CORREÇÃO: Checa movimentação real usando distância euclidiana (absoluta em todas as direções)
    distancia_percorrida = math.hypot(new_cord[0] - cord[0], new_cord[1] - cord[1])

    if distancia_percorrida > mov_limite:
        return True, area_max_1, new_cord

    return False, area_max_1, new_cord

# Inicializa câmera
camera = None
inicializar_camera()

def ler_frame_com_timeout(timeout=5):
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
    global camera # Precisamos da referência da câmera aqui para o replay/fps

    tempo_ultimo_frame = time.time()
    frames_sem_processar = 0

    # ========================================================
    # 1. DESCOBRE A VELOCIDADE ORIGINAL DO VÍDEO (FPS)
    # ========================================================
    fps_video = camera.get(cv2.CAP_PROP_FPS)
    if fps_video == 0 or math.isnan(fps_video):
        fps_video = 10.0 # Valor padrão de segurança

    atraso_por_frame = 1.0 / fps_video

    while executando_servidor:
        try:
            tempo_inicio_leitura = time.time() # Marca o tempo antes de ler

            # Lê frame com timeout
            frame = ler_frame_com_timeout(5)

            if frame is None:
                # ========================================================
                # 2. AUTO-REPLAY (Se o frame for None, o vídeo acabou!)
                # ========================================================
                print("🔄 Fim do vídeo! Reiniciando a gravação...")
                camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.5)
                continue

            # ========================================================
            # 3. FREIO DE MÃO (Simula o tempo real)
            # ========================================================
            tempo_gasto = time.time() - tempo_inicio_leitura
            tempo_espera = atraso_por_frame - tempo_gasto

            # Se leu o arquivo do PC muito rápido, dorme o tempo que falta
            if tempo_espera > 0:
                time.sleep(tempo_espera)

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
    global resetMaoPlayers, maos_jogadores
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
    global estado_intervalo, Zerou_mao, modoAuto

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

    # ====================================================================
    # 1. ENCONTRAR A SILHUETA SÓLIDA BASE
    # ====================================================================
     # 1. Máscara Sólida Base
    _, mask_branca = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
    contours_ext, _ = cv2.findContours(mask_branca, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("1 - Mask Branca", mask_branca)

    mask_solida = np.zeros_like(gray)
    cv2.drawContours(mask_solida, contours_ext, -1, 255, thickness=cv2.FILLED)

    # cv2.imshow("1 - Mask Solida", mask_solida)
    mask_solida = cv2.medianBlur(mask_solida, 5)
    # cv2.imshow("1 - Mask Solida Com Blur", mask_solida)

    global conf_busca, area_base, cord_cont, detectar_vales_por_morfologia
    global encontrar_pares_corte, cortar_nos_vales_inteligente

    # print(f"Valor de Coordenadas do contorno: {cord_cont}")

    processar = False
    if not conf_busca:
        processar, area_base, cord_cont = nova_pedra(mask_solida, CONFIG_VALES['area_min'], cord_cont)
        conf_busca = True
    else:
        processar, area_base, cord_cont = nova_pedra(mask_solida, area_base, cord_cont)
    # print(f"Valor de Coordenadas do contorno de averiguação: {cord_cont}")
    processar = True
    if processar:
        # Refinamento de Contornos
        cnts_pre, _ = cv2.findContours(mask_solida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_filtrada = np.zeros_like(gray)

        # Melhoria
        fator_area = zoom_factor ** 2
        area_min = int(CONFIG_VALES['area_min'] * fator_area)
        area_max = int(CONFIG_VALES['area_max'] * fator_area)
        raio_corte = int(CONFIG_VALES['distancia_corte'] * zoom_factor) # Distância é linear

        for c in cnts_pre:
            if cv2.contourArea(c) > area_min:
                cv2.drawContours(mask_filtrada, [c], -1, 255, -1)

        # Para remover os pontos pretos (subtrair áreas)
        mask_diferenca = cv2.bitwise_and(mask_filtrada, mask_branca)

        # Se preferir ver onde os pontos foram removidos:
        mask_pontos = cv2.bitwise_xor(mask_filtrada, mask_branca)

        pontos_vale = detectar_vales_por_morfologia(mask_filtrada)

        kernel_derreter = np.ones((7, 7), np.uint8)
        mask_corte = cv2.erode(mask_filtrada, kernel_derreter, iterations=3)


        contours_final, _ = cv2.findContours(mask_filtrada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnt_finais = []
        if len(pontos_vale) >= 2 and contours_final:
            # if args.debug:
            #     img_debug_final = visualizar_vales_detalhado(img, mask_filtrada, pontos_vale)

            # if args.debug:
            #     cv2.imshow("2 - Mask Usada", mask_filtrada)

            pares_corte = encontrar_pares_corte(pontos_vale, mask_filtrada, raio_corte)
            cnt_finais = cortar_nos_vales_inteligente(mask_filtrada, pontos_vale, pares_corte)
            max_contorno = max(cnt_finais, key=cv2.contourArea)
            min_contorno = min(cnt_finais, key=cv2.contourArea)
            max_contorno = cv2.contourArea(max_contorno)
            min_contorno = cv2.contourArea(min_contorno)
            # print(f"Maior contorno: {max_contorno}  -- Menor: {min_contorno}")

            if max_contorno > area_max * 1.8:
                # print("Recalculando com mascara reduzinda!")
                # print(f"Área max. permitida: {area_max * 1.8}  --  Área min: {area_min}")
                # print(f"Área do maior Contorno: {max_contorno}  --  Menor contorno: {min_contorno}")
                # contours_corte, _ = cv2.findContours(mask_corte, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # mask_corte = np.zeros_like(gray)
                # contorno_corte = max(contours_corte, key=cv2.contourArea)
                # cv2.drawContours(mask_corte, [contorno_corte], -1, 255, -1)

                pares_corte = encontrar_pares_corte(pontos_vale, mask_corte, raio_corte)
                cnt_finais = cortar_nos_vales_inteligente(mask_filtrada, pontos_vale, pares_corte)
                # if args.debug:
                #     cv2.imshow("2 - Mask Corte Usada", mask_corte)

            # if args.debug:
            #     visualizar_cortes(img_debug_final, mask_filtrada, mask_filtrada, cnt_finais, pares_corte, "2 - Cortes Aplicados")

        else:
            if contours_final:
                cnt_finais = contours_final
            # print("⚠️ Poucos vales detectados ou nenhum contorno encontrado!")

        global actions, zoom_change, automatic_zoom, zoom_prev

        candidatos = []

        if debug_mode:
            print(f"Contornos encontrados: {len(cnt_finais)}")

        for cnt in cnt_finais:
            area = cv2.contourArea(cnt)
            # if area < 10 or area > 2200:
            #     continue
            if area_max > area > area_max * 0.3:
                rect = cv2.minAreaRect(cnt)
                center, size, angle = rect
                w_box, h_box = size

                if w_box == 0 or h_box == 0:
                    # print("Box com dimensões zeradas")
                    continue

                width, height = size

                if width > height:
                    ratio = width/height
                    margem_A = 0.9
                    margem_L = 1.02

                else:
                    ratio = height/width
                    margem_A = 1.05
                    margem_L = 0.95

                # print(f"Valor de ratio: {ratio} e Área: {area}")

                if not (2.4 >= ratio >= 1.5):
                    # print(f"Ratio fora do padrão: {ratio}")
                    continue

                rect_pedra = (center, (width*margem_L, height*margem_A), angle)

                candidatos.append({
                    'rect_pedra': rect_pedra,
                    'centro': center,
                    'area': area,
                    'ratio': ratio,
                })

        if debug_mode:
            print(f"Candidatos aprovados: {len(candidatos)}")

        # Filtro por distância
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


        # Configuração de bando
        DISTANCIA_CONEXAO = CONFIG_VALES['distancia_conexao'] * zoom_factor
        agrupamento = True
        pedras_aprovadas = []

        if agrupamento:
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
        else:
            pedras_aprovadas = candidatos


        # Ordena as pedras de cima para baixo (pelo eixo Y do centro)
        pedras_aprovadas.sort(key=lambda x: x['centro'][1])

        # =================================================================
        # --- LEITURA DIRETA E PREPARAÇÃO PARA A WEB ---
        # =================================================================
        # Protegido por "if processar", este bloco só roda quando a mesa muda.
        # Dispensamos a lógica de fantasmas e o cache individual.

        lista_final = []
        pedras_vistas_agora = []

        for d in pedras_aprovadas[:]:
            cx_nova, cy_nova = d['centro']

            # Lemos os valores reais direto da imagem cortada sem depender de cache
            pts_cima, pts_baixo, zero = extrair_e_contar(mask_pontos, d['rect_pedra'])
            valor_pedra = f"{pts_cima}|{pts_baixo}"

            if valor_pedra == "0|0" and not zero:
                pedras_aprovadas.remove(d)
                continue

            # Bloqueio de leitura dupla no mesmo frame (ou nas mãos dos jogadores)
            if not valor_ja_existe(valor_pedra, modo_leitura, pedras_vistas_agora):
                pedras_vistas_agora.append({'valor': valor_pedra})

                # --- INTELIGÊNCIA DE ORIENTAÇÃO (CSS) ---
                w, h = d['rect_pedra'][1]
                angulo_cv = d['rect_pedra'][2]

                # O CSS desenha as pedras "em pé" por padrão (0 graus).
                if w < h:
                    angulo_corrigido = angulo_cv - 90
                else:
                    angulo_corrigido = angulo_cv

                lista_final.append({
                    'valor': valor_pedra,
                    'x': cx_nova,
                    'y': cy_nova,
                    'angulo': angulo_corrigido
                })
            else:
                # Imprime rejeições apenas nas mãos para não poluir o terminal da mesa
                if modo_leitura != 'mesa':
                    print(f"🚫 Duplicata rejeitada: {valor_pedra}")
                    continue

        # ONDE ESTAMOS A OLHAR?
        if modo_leitura == 'mesa':
            ultima_leitura_pedras = lista_final
        else:
            # Na leitura da mão, só validamos se houverem exatamente 7 pedras
            if len(lista_final) == 7:
                maos_jogadores[modo_leitura] = lista_final
                print(f"✅ Mão de {modo_leitura} atualizada com {len(lista_final)} pedras")

        # =================================================================
        # --- PREPARA FRAME PARA STREAMING (WEBCAM/VIDEO) ---
        # =================================================================
        if enviar_video:
            out = img.copy()

            # Usamos a lista_final (que já tem o ângulo e o valor corrigidos para a Web)
            # ou a pedras_aprovadas (que tem as caixas retangulares cruas do OpenCV).
            # Como você quer desenhar o rect_pedra, vamos usar o pedras_aprovadas original daquele frame.

            for p in pedras_aprovadas:  # Limita desenho a 20 pedras por performance
                try:
                    # # Desenha o traço central (fenda) em azul
                    # box_traco = np.int32(cv2.boxPoints(p['rect_traco']))
                    # cv2.drawContours(out, [box_traco], 0, (255, 0, 0), 1)

                    # Desenha a caixa principal da pedra em verde
                    box_pedra = np.int32(cv2.boxPoints(p['rect_pedra']))
                    cv2.drawContours(out, [box_pedra], 0, (0, 255, 0), 1)

                    # Opcional (Recomendado): Escrever o valor lido na tela do stream para debug visual
                    cx, cy = map(int, p['centro'])
                    # Como tiramos o valor de pedras_aprovadas, precisamos pegar da leitura.
                    # Se você preferir não ler o valor aqui para poupar CPU, basta remover as linhas abaixo.
                    pts_cima, pts_baixo, zero = extrair_e_contar(mask_pontos, p['rect_pedra'])
                    area = p['area']
                    ratio = p['ratio']
                    if not zero and f"{pts_cima}|{pts_baixo}" == "0|0":
                        continue
                    if debug_mode:
                        cv2.putText(out, f"{ratio:.2f}", (cx - 30, cy - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
                        cv2.putText(out, f"{ratio:.2f}", (cx - 30, cy - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.putText(out, f"{int(area)}", (cx - 30, cy - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
                        cv2.putText(out, f"{int(area)}", (cx - 30, cy - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(out, f"{pts_cima}|{pts_baixo}", (cx - 15, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                    cv2.putText(out, f"{pts_cima}|{pts_baixo}", (cx - 15, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                except Exception as e:
                    # Boa prática: imprimir o erro no terminal ajuda a debugar se algo falhar
                    # print(f"Erro ao desenhar contorno no stream: {e}")
                    pass

            # Codifica a imagem para JPEG com compressão de 70% (bom equilíbrio tamanho/qualidade)
            sucesso_encode, buffer = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if sucesso_encode:
                ultimo_frame_processado = buffer.tobytes()
        else:
            ultimo_frame_processado = None

# ====================================================================
# SUBSISTEMA DE LOCALIZAÇÂO DE VALES
# ====================================================================

def detectar_vales_por_morfologia(mask_solida):
    """
    Aplica a ideia de preencher as fendas e subtrair a imagem original
    para isolar os vales.
    """
    # 1. 'Massa Corrida' (Fechamento)
    k_size = CONFIG_VALES['tamanho_kernel_morfologia']
    kernel_fechamento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

    mask_fechada = cv2.morphologyEx(mask_solida, cv2.MORPH_CLOSE, kernel_fechamento)

    # 2. Subtração (O Pulo do Gato)
    mask_vales = cv2.subtract(mask_fechada, mask_solida)
    # if args.debug:
    #     cv2.imshow("3 - Mask Vales", mask_vales) # Descomente se precisar debugar

    # 3. Extrair os Pontos
    cnts_vales, _ = cv2.findContours(mask_vales, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pontos_encontrados = []
    for c in cnts_vales:
        cx, cy = cv2.minAreaRect(c)[0]
        area = cv2.contourArea(c)
        # Correção: passar como uma tupla contendo a coordenada e a área
        pontos_encontrados.append(([cx, cy], area))

    return agrupar_pontos_proximos(pontos_encontrados, int(CONFIG_VALES['distancia_filtro'] * zoom_factor))

def agrupar_pontos_proximos(dados_pontos, raio=5):
    """
    Recebe uma lista no formato [ ([cx, cy], area), ... ]
    Ordena por área para garantir a sobrevivência do ponto mais forte
    sem perder a performance do NumPy.
    """
    if len(dados_pontos) == 0:
        return np.array([])

    # 1. O Pulo do Gato: Ordenar do maior para o menor (pela área)
    # Assim, o primeiro ponto de qualquer aglomeração SEMPRE será o "mais forte"
    dados_ordenados = sorted(dados_pontos, key=lambda x: x[1], reverse=True)

    # 2. Separar apenas as coordenadas para a matemática vetorial do NumPy
    pontos = np.array([item[0] for item in dados_ordenados])

    finais = []
    visitados = np.zeros(len(pontos), dtype=bool)

    for i in range(len(pontos)):
        if visitados[i]:
            continue

        # Como ordenamos antes, este p_atual é garantidamente o de MAIOR ÁREA na vizinhança
        p_atual = pontos[i]
        finais.append(p_atual.astype(int))

        # Magia do NumPy: Calcula a distância deste ponto para TODOS os outros de uma vez
        distancias = np.linalg.norm(pontos - p_atual, axis=1)

        # Marca como 'visitado' (descarta) todos que estiverem dentro do raio de tolerância.
        # Os que estão sendo descartados têm área menor ou igual ao p_atual.
        visitados[distancias < raio] = True

    return np.array(finais)
# ====================================================================
# SUBSISTEMA DE CORTES REFATORADO E OTIMIZADO
# ====================================================================

def encontrar_pares_corte(pontos_vale, mask_pedras, raio_max=69):
    """
    Vetorizado. Usa a máscara sólida em vez de polígonos para evitar
    o problema de pedras isoladas (ilhas) sendo ignoradas.
    """
    if len(pontos_vale) < 2:
        return []

    pontos = np.array(pontos_vale)
    n_pontos = len(pontos)

    diffs = pontos[:, np.newaxis, :] - pontos[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diffs, axis=-1)

    pares_candidatos = []
    altura_img, largura_img = mask_pedras.shape[:2]

    for i in range(n_pontos):
        for j in range(i + 1, n_pontos):
            dist = dist_matrix[i, j]

            if dist < raio_max:
                # 2. NOVA Verificação Rápida e Robusta de Interseção:
                # Testamos 3 pontos internos ao longo da linha (25%, 50% e 75%)
                # para evitar falsos negativos caso a pedra tenha bordas irregulares.

                pA = pontos[i]
                pB = pontos[j]

                # Fatores de interpolação (o quão longe estamos de A em direção a B)
                fracoes = [0.25, 0.50, 0.75]

                linha_valida = False
                for f in fracoes:
                    # Calcula o ponto exato naquela fração da linha
                    pt_amostra = pA + (pB - pA) * f
                    mx, my = int(pt_amostra[0]), int(pt_amostra[1])

                    # Checagem de segurança dos limites da imagem
                    if 0 <= my < altura_img and 0 <= mx < largura_img:
                        # Se achou pelo menos um pixel branco forte, a linha cruza a pedra
                        if mask_pedras[my, mx] > 0:
                            linha_valida = True
                            break # Otimização: não precisa testar as outras frações

                # Se após testar os 3 pontos todos caíram no fundo preto, descarta o par.
                if not linha_valida:
                    continue

                # Pontuação base (distância)
                score = 100.0 / (1.0 + dist)

                # 3. Triangulação (Identificar ângulos +- 90°)
                bonus_triangulacao = 1.0
                # dist_AB = dist

                for k in range(n_pontos):
                    if k != i and k != j:
                        dist_AC = dist_matrix[i, k]

                        # Usando a sua margem de proporção testada
                        dist_real = 65 * zoom_factor
                        # if (dist_AB * 1.7) < dist_AC < (dist_AB * 3.2):
                        if dist_real > dist_AC > dist_real * 0.5:
                            # print(f"Corte Dentro do range: dist_AB: {dist_AB} -- dist_AC: {dist_AC}")
                            v1 = pontos[j] - pontos[i]
                            v2 = pontos[k] - pontos[i]
                            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)

                            if n1 > 0 and n2 > 0:
                                cos_theta = np.dot(v1, v2) / (n1 * n2)
                                if abs(cos_theta) < 0.35:
                                    bonus_triangulacao = 2.0
                                    break
                #         # else:
                        #     print(f"Corte fora do range: dist_AB: {dist_AB} -- dist_AC: {dist_AC}")
                score *= bonus_triangulacao
                pares_candidatos.append((i, j, score))

    # Ordenar pelos melhores cortes
    pares_candidatos.sort(key=lambda x: x[2], reverse=True)

    # Evitar reutilização de pontos
    pares_finais = []
    pontos_usados = set()

    for i, j, score in pares_candidatos:
        if i not in pontos_usados and j not in pontos_usados:
            pares_finais.append((pontos[i], pontos[j], score))
            pontos_usados.add(i)
            pontos_usados.add(j)

    return pares_finais

def cortar_nos_vales_inteligente(mask_pedra_solida, pontos_vale, pares_corte):
    """
    Aplica as linhas de corte geradas pelo algoritmo otimizado.
    """
    if not pares_corte:
        contours, _ = cv2.findContours(mask_pedra_solida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    mask_cortada = mask_pedra_solida.copy()

    for p1, p2, _ in pares_corte:
        cv2.line(mask_cortada, p1, p2, 0, thickness=2)
        cv2.circle(mask_cortada, p1, 2, 0, -1)
        cv2.circle(mask_cortada, p2, 2, 0, -1)

    contours_apos, _ = cv2.findContours(mask_cortada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours_apos



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
    global INTERVALO_SEGUNDOS
    dados = request.get_json()

    INTERVALO_SEGUNDOS = float(dados.get('intervalo_segundos', INTERVALO_SEGUNDOS))
    if dados:
        print(f"Ataulizado o tempo de leitura - {INTERVALO_SEGUNDOS}")
    # Atualiza as variáveis globais em tempo real!


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

    if actions['action1'] == 'calibrar':
        modoAuto = True

    if actions['action2'] == 'valor_zoom':
        return jsonify({"zoom": zoom_factor})

    return jsonify({"status": "sucesso"})

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
