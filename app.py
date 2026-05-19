from flask import Flask, jsonify, render_template, request, Response, send_file
import cv2
import numpy as np
import math
import time
import threading
import atexit
import argparse
from collections import deque

app = Flask(__name__)

# Configurações otimizadas
# device = 'http://192.168.0.129:5000/video?video_size=1920x1080'
# device = '/home/rivelino/Downloads/rec_2026-04-07_21-49.mp4'
# device = '/sdcard/Movies/IPcam/rec_2026-05-12_23-04.mp4'
# device = '/home/rivelino/Downloads/rec_2026-05-12_23-04.mp4'
# device = '/home/rivelino/Downloads/rec_2026-04-20_00-17.mp4'
# device = '/home/rivelino/Git/project_game_domino/teste_colocamento_de_pedras.mp4'
zoom_factor = 0.0
ultima_leitura_pedras = []
ultimo_tempo_processamento = 0
ultimo_frame_processado = None
INTERVALO_SEGUNDOS = 0.5
executando_servidor = True
enviar_video = True
DISTANCIA_MINIMA = 37
modo_leitura = 'mesa'
actions = {'rst': 'False', 'zoom': '0.0'}
resetMaoPlayers = False
tirar_foto_debug = False
maos_jogadores = {'p1': [], 'p2': [], 'p3': [], 'p4': []}
Zerou_mao = False
duplicada = None
zoom_reset = False
largura_frame = 0
altura_frame = 0
ultimo_processamento_forcado = 0.0

debug_mode = True
start = True

# Cache para frames para evitar processamento repetido
frame_buffer = deque(maxlen=1)
ultimo_frame_valido = None
falhas_consecutivas = 0
MAX_FALHAS = 10
conf_busca = False
cord_cont = (0, 0)
area_base = 0


def parse_arguments():
    parser = argparse.ArgumentParser(description='Processa imagens de dominó')
    parser.add_argument('arquivo', help='Caminho do arquivo ou link da câmera.')
    parser.add_argument('-z', '--zoom', type=float, default=1.4, help='Nível de zoom (padrão 1.0)')
    parser.add_argument('-p', '--proximidade', type=int, default=37, help='Distância mínima entre pedras')
    parser.add_argument('-L', '--limiar', type=int, default=190, help='Limiar de branco (0-255), valores de uso 150-200')
    parser.add_argument('-d', '--debug', action='store_true', help='Ativa modo depuração')
    return parser.parse_args()

device = parse_arguments().arquivo

CONFIGS = {
    'distancia_filtro': 15,
    'distancia_mov': 15,
    'distancia_corte': 62,
    'tamanho_kernel_morfologia': 13, # Novo parâmetro para o tamanho da fenda a ser fechada
    'area_max': 4000,                # Area maxima das pedras
    'area_min': 500,
    'area_ponto': 12,
    'distancia_conexao': 200,
}


##  Extração dos pontos das pedras ######
def extrair_e_contar(img, rect_pedra):
    center, size, angle = rect_pedra
    cx, cy = center

    w, h = size
    if w > h:
        w, h = h, w
        angle += 90

    w_int = int(round(w))
    h_int = int(round(h))

    # --- Rotaciona com INTER_NEAREST para preservar binário sem blur ---
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    altura_img, largura_img = img.shape[:2]
    img_rot = cv2.warpAffine(img, M, (largura_img, altura_img),
                             flags=cv2.INTER_NEAREST)  # <-- SEM interpolação

    # Recorta exatamente o bounding rect
    x1 = max(0, int(round(cx - w_int / 2)))
    y1 = max(0, int(round(cy - h_int / 2)))
    x2 = min(largura_img, x1 + w_int)
    y2 = min(altura_img, y1 + h_int)

    pedra_recortada = img_rot[y1:y2, x1:x2]

    if pedra_recortada.size == 0:
        return 0, 0, False, 0.0

    # Verifica se tem conteúdo (zero|zero real vs pedra branca pura)
    contornos_total, _ = cv2.findContours(pedra_recortada, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)

    # ----------------------------------------------------------------
    # LOCALIZAR A FENDA (divisor real entre as metades)
    # ----------------------------------------------------------------
    meio, zero_local = _encontrar_fenda(pedra_recortada)

    metade_cima = pedra_recortada[0:meio, :]
    metade_baixo = pedra_recortada[meio:, :]

    med_ar = 0.0
    count = 0
    med_area_bruta = 0.0

    for c in contornos_total:
        area = cv2.contourArea(c)
        perimetro = cv2.arcLength(c, True)
        if perimetro == 0:
            continue
        circularidade = 4 * np.pi * (area / (perimetro * perimetro))
        if circularidade >= 0.5:
            med_ar += area
            count += 1

    if med_ar > 0:
        med_area_bruta = med_ar / count

    pts_cima, med_ar1 = contar_bolinhas(med_area_bruta, metade_cima)
    pts_baixo, med_ar2 = contar_bolinhas(med_area_bruta, metade_baixo)

    med_area = 0.0
    if med_ar1 > 0.0 or med_ar2 > 0.0:
        med_area = abs((abs(med_ar1) + abs(med_ar2)) / 2)

    return pts_cima, pts_baixo, zero_local, med_area

def _encontrar_fenda(pedra_bin):
    h_total = pedra_bin.shape[0]
    w_total = pedra_bin.shape[1]

    margem = h_total // 4
    zona = pedra_bin[margem: h_total - margem, :]

    # SEM inverter — procura contornos brancos na zona central
    # A fenda deixa bordas brancas finas acima e abaixo dela
    # que formam retângulos longos e finos (ratio alto)
    contornos, _ = cv2.findContours(zona, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    melhor_ratio = 0.0
    melhor_y = None
    largura_minima = w_total * 0.3
    zero_local = False

    for c in contornos:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch == 0 or cw == 0:
            continue

        # Só interessa retângulos MAIS LARGOS que altos (fenda = comprida e fina)
        if cw <= ch:
            continue

        ratio = cw / ch

        if ratio > melhor_ratio and cw >= largura_minima:
            melhor_ratio = ratio
            melhor_y = margem + y + ch // 2
        if ratio > 3.0:
            zero_local = True

    if melhor_y is None:
        return h_total // 2, zero_local

    return melhor_y, zero_local

def contar_bolinhas(med_bruta, metade):
    """
    Conta bolinhas numa metade. Recebe med_bruta para
    poder calibrar area_ponto dinamicamente.
    """
    contornos, _ = cv2.findContours(metade, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    pontos = 0
    med_area = 0.0

    for c in contornos:
        area = cv2.contourArea(c)
        # if point_area * 0.4 < area < point_area * 2.5:
        perimetro = cv2.arcLength(c, True)
        if perimetro == 0:
            continue
        circularidade = 4 * np.pi * (area / (perimetro * perimetro))
        # print(f"Circularidade: {circularidade}")
        if circularidade >= 0.5 and (med_bruta * 2.2) >= area >= (med_bruta * 0.1) :   # levemente mais permissivo pós INTER_NEAREST
            med_area += area
            pontos += 1

    if med_area > 0.0 and pontos > 0:
        med_area = abs(med_area / pontos)

    return min(pontos, 6), med_area


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

        if device.startswith(("http://", "https://")):
            camera = cv2.VideoCapture(device)
        else:
            camera = cv2.VideoCapture(device, cv2.CAP_FFMPEG)

        if not camera.isOpened():
            print("⚠️ Falha ao abrir câmera, tentando novamente...")
            time.sleep(1)
            if device.startswith(("http://", "https://")):
                camera = cv2.VideoCapture(device)
            else:
                camera = cv2.VideoCapture(device, cv2.CAP_FFMPEG)


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
        return False, 0, (0, 0), 0.0

    maior_cnt = max(contours_1, key=cv2.contourArea)
    area_max_1 = cv2.contourArea(maior_cnt)

    fator_area = zoom_factor ** 2
    area_test = CONFIGS['area_min'] * fator_area

    # minAreaRect retorna (centro(x,y), tamanho(w,h), angulo)
    new_cord, _, _ = cv2.minAreaRect(maior_cnt)

    mov_limite = CONFIGS['distancia_mov'] * zoom_factor

    if not area_max_1 or not area_max_2:
        return False, 0, (0, 0), 0.0 # CORREÇÃO: removido o tuple(0, 0)

    # Checa alteração brusca de área (Pedra entrou ou saiu da mesa)
    if area_max_1 > (area_max_2 + area_test) or area_max_1 < (area_max_2 - area_test):
        return True, area_max_1, new_cord, time.time()

    # CORREÇÃO: Checa movimentação real usando distância euclidiana (absoluta em todas as direções)
    distancia_percorrida = math.hypot(new_cord[0] - cord[0], new_cord[1] - cord[1])

    if distancia_percorrida > mov_limite:
        return True, area_max_1, new_cord, time.time()

    return False, area_max_1, new_cord, 0.0

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
                processar_frame(frame, tempo_atual, args)

            # Só envia vídeo se necessário
            if enviar_video and ultimo_frame_processado is not None:
                # Limita FPS do stream para não sobrecarregar
                if time.time() - tempo_ultimo_frame > 0.1:
                    tempo_ultimo_frame = time.time()
            else:
                # Pequena pausa para não consumir CPU
                time.sleep(0.1)

        except Exception as e:
            print(f"Erro no loop principal: {e}")
            time.sleep(0.5)

def corrigir_orientacao(img):
    """
    Garante que a imagem sempre saia em modo PAISAGEM (largura > altura).
    Independente de como o celular estava segurado ou da linha rotate.
    """
    h, w = img.shape[:2]
    if h > w:
        # Imagem em portrait → rotaciona para landscape
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img

def validar_rect_na_mask(rect_pedra, mask_filtrada, limiar_ocupacao=0.80):
    """
    Verifica se o rect_pedra cobre ao menos limiar_ocupacao (80%) de pixels
    brancos na mask_filtrada. Retorna True se for uma pedra válida.
    """
    # Cria máscara do rect rotacionado
    mask_rect = np.zeros(mask_filtrada.shape, dtype=np.uint8)
    box = np.int32(cv2.boxPoints(rect_pedra))
    cv2.fillPoly(mask_rect, [box], 255)

    # Pixels dentro do rect
    total_pixels = cv2.countNonZero(mask_rect)
    if total_pixels == 0:
        return False

    # Pixels brancos na mask_filtrada dentro do rect
    intersecao = cv2.bitwise_and(mask_filtrada, mask_rect)
    pixels_brancos = cv2.countNonZero(intersecao)

    ocupacao = pixels_brancos / total_pixels
    return ocupacao >= limiar_ocupacao


def processar_frame(img, tempo_atual, args):
    """Processa o frame de forma otimizada"""
    global ultima_leitura_pedras, ultimo_frame_processado, duplicada
    global resetMaoPlayers, maos_jogadores, start
    global modo_leitura, tirar_foto_debug, enviar_video, zoom_factor
    global largura_frame, altura_frame

    ## Variaveis
    debug_mode = args.debug
    if start:
        zoom_factor = args.zoom
        start = False

    # Rotaciona a imagem para ficar mais adequando à mesa
    img = corrigir_orientacao(img)

    # Aplica zoom se necessário
    if zoom_factor != 1.0:
        img = cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor,
                        interpolation=cv2.INTER_LINEAR)

    largura_frame = img.shape[1]
    altura_frame = img.shape[0]

    # Reset das mãos se necessário
    if resetMaoPlayers:
        print("Resetando mãos dos jogadores...")
        resetMaoPlayers = False
        for player in maos_jogadores:
            maos_jogadores[player] = []

    # Configura intervalos dinâmicos
    global Zerou_mao, DISTANCIA_MINIMA

    if modo_leitura != 'mesa':
        DISTANCIA_MINIMA = 25

        if not Zerou_mao:
            maos_jogadores[modo_leitura] = []
            Zerou_mao = True
    else:
        DISTANCIA_MINIMA = 37
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
    _, mask_branca = cv2.threshold(gray, args.limiar, 255, cv2.THRESH_BINARY)
    contours_ext, _ = cv2.findContours(mask_branca, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("1 - Mask Branca", mask_branca)

    mask_solida = np.zeros_like(gray)
    cv2.drawContours(mask_solida, contours_ext, -1, 255, thickness=cv2.FILLED)

    # cv2.imshow("1 - Mask Solida", mask_solida)
    # mask_solida = cv2.medianBlur(mask_solida, 5)
    # cv2.imshow("1 - Mask Solida Com Blur", mask_solida)

    global conf_busca, area_base, cord_cont, detectar_vales_por_morfologia
    global encontrar_pares_corte, cortar_nos_vales_inteligente
    global ultimo_processamento_forcado

    # print(f"Valor de Coordenadas do contorno: {cord_cont}")
    # out = img.copy()
    out = None

    processar = False

    # 1. Forçar processamento mínimo a cada N segundos mesmo sem movimento
    INTERVALO_FORCADO = 5.0  # segundos

    # processar, area_base, cord_cont, time_exec = nova_pedra(mask_solida, area_base, cord_cont)



    if not conf_busca:
        processar, area_base, cord_cont, time_exec = nova_pedra(mask_solida, CONFIGS['area_min'], cord_cont)
        conf_busca = True
    else:
        processar, area_base, cord_cont, time_exec = nova_pedra(mask_solida, area_base, cord_cont)
    # print(f"Valor de Coordenadas do contorno de averiguação: {cord_cont}")
    # processar = True

    tempo_atual_local = time.time()
    if not processar and (tempo_atual_local - ultimo_processamento_forcado) > INTERVALO_FORCADO:
        processar = True
        ultimo_processamento_forcado = tempo_atual_local

    if processar or time.time() - time_exec <= 16.5:
        # Refinamento de Contornos
        cnts_pre, _ = cv2.findContours(mask_solida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_filtrada = np.zeros_like(gray)

        # Melhoria
        fator_area = (zoom_factor - 0.4) ** 2
        area_min = int(CONFIGS['area_min'] * fator_area)
        area_max = int(CONFIGS['area_max'] * fator_area)
        raio_corte = int(CONFIGS['distancia_corte'] * zoom_factor) # Distância é linear

        for c in cnts_pre:
            if cv2.contourArea(c) > area_min:
                cv2.drawContours(mask_filtrada, [c], -1, 255, -1)

        # Alinhar contorno
        def alinhar_contorno(contorno):
            """
            Rotaciona o contorno pelo ângulo do minAreaRect
            para que o lado longo fique alinhado com o eixo Y (vertical).
            Retorna o boundingRect ajustado.
            """
            rect = cv2.minAreaRect(contorno)
            center, size, angle = rect
            w, h = size

            # Garante que h é sempre o lado longo
            if w > h:
                w, h = h, w
                angle += 90

            # Rotaciona os pontos do contorno em torno do centro
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Aplica a rotação nos pontos do contorno
            pontos = contorno.reshape(-1, 2).astype(np.float32)
            pontos_rot = cv2.transform(pontos.reshape(1, -1, 2), M).reshape(-1, 2)

            # boundingRect agora é ajustado ao eixo
            x = int(pontos_rot[:, 0].min())
            y = int(pontos_rot[:, 1].min())
            cw = int(pontos_rot[:, 0].max()) - x
            ch = int(pontos_rot[:, 1].max()) - y

            return x, y, cw, ch, angle

        # Se preferir ver onde os pontos foram removidos:
        mask_base_pontos = cv2.bitwise_xor(mask_filtrada, mask_branca)
        cnts_pontos, _ = cv2.findContours(mask_base_pontos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_esp_pontos = np.zeros_like(gray)
        mask_esp_tracos = np.zeros_like(gray)

        # Obter as médias
        med_cw_tracos = 0.0
        counter = 0

        for c in cnts_pontos:
            x, y, cw, ch, angle = alinhar_contorno(c)
            ratio = ch / cw if cw > 0 else 0
            if ratio > 2.5:
                med_cw_tracos += cw
                counter += 1

        med_cw_tracos = med_cw_tracos / counter if med_cw_tracos > 0 else 0

        for c in cnts_pontos:
            x, y, cw, ch, angle = alinhar_contorno(c)

            ratio = ch / cw if cw > 0 else 0

            if ratio > 2.5 and cw < med_cw_tracos + 2:
                cv2.drawContours(mask_esp_tracos, [c], -1, 255, -1)
            else:
                cv2.drawContours(mask_esp_pontos, [c], -1, 255, -1)

        cnts_tracos, _ = cv2.findContours(
            mask_esp_tracos,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        def angle_diff(a, b):
            d = abs(a - b) % 180
            return min(d, 180 - d)

        fragmentos = []

        for c in cnts_tracos:

            x, y, cw, ch, angle = alinhar_contorno(c)

            cx = x + cw // 2
            cy = y + ch // 2

            fragmentos.append({
                'contour': c,
                'x': x,
                'y': y,
                'w': cw,
                'h': ch,
                'cx': cx,
                'cy': cy,
                'angle': angle
            })

        mask_tracos_unidos = np.zeros_like(gray)

        for frag in fragmentos:
            cv2.drawContours(
                mask_tracos_unidos,
                [frag['contour']],
                -1,
                255,
                -1
            )

        for i in range(len(fragmentos)):

            for j in range(i + 1, len(fragmentos)):

                a = fragmentos[i]
                b = fragmentos[j]

                # diferença vertical pequena
                dy = abs(a['cy'] - b['cy'])

                # distância horizontal
                dx = abs(a['cx'] - b['cx'])

                # ângulo parecido
                da = angle_diff(
                    a['angle'],
                    b['angle']
                )

                if (
                    dy < 25
                    and dx < 25
                    and da < 15
                ):

                    pt1 = (a['cx'], a['cy'])
                    pt2 = (b['cx'], b['cy'])

                    cv2.line(
                        mask_tracos_unidos,
                        pt1,
                        pt2,
                        255,
                        thickness=1
                    )

        # ===========================================================
        # DISTANCE TRANSFORM
        # ============================================================

        # Garantir imagem binária
        mask_bin = (mask_esp_pontos > 0).astype(np.uint8)

        # Distance transform
        dist = cv2.distanceTransform(
            mask_bin,
            cv2.DIST_L2,
            3
        )

        # Threshold dos picos
        _, mask_pontos_sep = cv2.threshold(
            dist,
            0.38 * dist.max(),
            255,
            cv2.THRESH_BINARY
        )

        mask_pontos_sep = np.uint8(mask_pontos_sep)

        # Pequena dilatação para recuperar formato
        kernel_restore = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )

        mask_pontos_sep = cv2.dilate(
            mask_pontos_sep,
            kernel_restore,
            iterations=1
        )
        mask_pontos = cv2.bitwise_or(mask_tracos_unidos, mask_pontos_sep)

        if debug_mode:
            # Converte binário para BGR (3 canais)
            out = cv2.cvtColor(mask_pontos, cv2.COLOR_GRAY2BGR)
        else:
            out = img.copy()

        # --- Coleta contornos dos traços e calcula média do comprimento ---
        cnts_tracos, _ = cv2.findContours(mask_tracos_unidos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        comprimentos = []
        for c in cnts_tracos:
            r = cv2.minAreaRect(c)
            _, (w_t, h_t), _ = r
            comp = max(w_t, h_t)
            esp  = min(w_t, h_t)
            if esp == 0:
                continue
            ratio_t = comp / esp
            # Traço válido: comprido e fino
            if ratio_t > 3.0 and comp > 8 * args.zoom:
                comprimentos.append(comp)

        if comprimentos:
            med_comp_traco = float(np.median(comprimentos))

            # Pedra de dominó: traço central ≈ 85% da largura interna da pedra
            # Ratio pedra ≈ 2:1  →  altura_pedra ≈ 2 * largura_pedra
            # largura_pedra ≈ comp_traco / 0.85
            # altura_pedra  ≈ largura_pedra * 2
            largura_pedra_est = med_comp_traco / 0.80
            altura_pedra_est  = largura_pedra_est * 2.0

            # Margem extra para não cortar as bolinhas nas bordas
            margem_extra = 1.06
            largura_final = largura_pedra_est * margem_extra
            altura_final  = altura_pedra_est  * margem_extra

            candidatos = []

            for c in cnts_tracos:
                r = cv2.minAreaRect(c)
                (cx_t, cy_t), (w_t, h_t), angle_t = r
                comp = max(w_t, h_t)
                esp  = min(w_t, h_t)
                if esp == 0:
                    continue
                ratio_t = comp / esp

                if ratio_t < 3.0 or not (med_comp_traco * 0.6 <= comp <= med_comp_traco * 1.4):
                    continue

                # Usa alinhar_contorno para obter o ângulo real do lado longo
                # independente de como o minAreaRect ordenou w/h
                _, _, cw_al, ch_al, angle_alinhado = alinhar_contorno(c)
                # angle_alinhado já aponta para o lado longo (ch > cw após alinhamento)
                # rect_pedra: largura_final é o lado curto, altura_final é o lado longo
                # o ângulo do minAreaRect para o lado longo = angle_alinhado - 90
                angle_pedra = angle_alinhado - 90

                rect_pedra_traco = ((cx_t, cy_t), (largura_final, altura_final), angle_pedra)

                if not validar_rect_na_mask(rect_pedra_traco, mask_filtrada):
                    continue  # rect mal orientado ou fora da pedra real

                candidatos.append({
                    'rect_pedra': rect_pedra_traco,
                    'centro': (cx_t, cy_t)
                })

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
            DISTANCIA_CONEXAO = CONFIGS['distancia_conexao'] * zoom_factor
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
            med_area_ponto = 0.0

            for d in pedras_aprovadas[:]:
                cx_nova, cy_nova = d['centro']

                # Lemos os valores reais direto da imagem cortada sem depender de cache
                pts_cima, pts_baixo, zero, med_ar = extrair_e_contar(mask_pontos, d['rect_pedra'])
                valor_pedra = f"{pts_cima}|{pts_baixo}"
                med_area_ponto += med_ar

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
            if med_area_ponto > 0.0:
                med_area_ponto = abs(med_area_ponto / len(pedras_aprovadas))

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
                # Usamos a lista_final (que já tem o ângulo e o valor corrigidos para a Web)
                # ou a pedras_aprovadas (que tem as caixas retangulares cruas do OpenCV).
                # Como você quer desenhar o rect_pedra, vamos usar o pedras_aprovadas original daquele frame.

                cv2.rectangle(out, (5,5), (500,200), (120,120,120), -1)
                cv2.putText(out, f"Dimensoes da imagem processada: {largura_frame}x{altura_frame}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(out, f"Area media dos Pontos: {med_area_ponto:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

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
                        pts_cima, pts_baixo, zero, _ = extrair_e_contar(mask_pontos, p['rect_pedra'])

                        if not zero and f"{pts_cima}|{pts_baixo}" == "0|0":
                            continue
                        cv2.putText(out, f"{pts_cima}|{pts_baixo}", (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                        cv2.putText(out, f"{pts_cima}|{pts_baixo}", (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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
    k_size = CONFIGS['tamanho_kernel_morfologia']
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

    return agrupar_pontos_proximos(pontos_encontrados, int(CONFIGS['distancia_filtro'] * zoom_factor)), mask_vales

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

def cortar_nos_vales_inteligente(mask_pedra_solida, img_debug, pontos_vale, pares_corte):
    """
    Aplica as linhas de corte geradas pelo algoritmo otimizado.
    """
    if not pares_corte:
        contours, _ = cv2.findContours(mask_pedra_solida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    mask_cortada = mask_pedra_solida.copy()

    for p1, p2, _ in pares_corte:
        cv2.line(mask_cortada, p1, p2, 0, thickness=2)
        cv2.circle(mask_cortada, p1, 3, 0, -1)
        cv2.circle(mask_cortada, p2, 3, 0, -1)
        if debug_mode:
            cv2.circle(img_debug, p1, 4, (1,1,255), 1)
            cv2.circle(img_debug, p2, 4, (1,1,255), 1)

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

@app.route('/favicon.ico')
def icon():
    return send_file('favicon.ico', mimetype='image/x-icon')

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
    global actions, resetMaoPlayers, zoom_factor
    dados = request.get_json()
    actions['rst'] = dados.get('reset')
    actions['zoom'] = dados.get('get_zoom')

    # Se fomos ler a mão de alguém, armamos o gatilho da foto!
    print(f"Valor do POST reset: {actions['rst']}  --  Valor de zoom: {actions['zoom']}")

    if actions['rst']:
        resetMaoPlayers = True

    if actions['zoom']:
        return jsonify({"zoom": zoom_factor})

    return jsonify({"status": "sucesso"})

@app.route('/api/estado_jogo')
def estado_jogo():
    # Esta rota envia TUDO (mesa e jogadores) para o HTML desenhar de uma vez só
    return jsonify({
        "modo_atual": modo_leitura,
        "mesa": ultima_leitura_pedras,
        "jogadores": maos_jogadores,
        "duplicada": duplicada,
        "frame_largura": largura_frame,
        "frame_altura": altura_frame
    })

if __name__ == '__main__':
    # Inicia o loop da câmera em uma thread separada para não travar o servidor Web!
    args = parse_arguments()
    t = threading.Thread(target=loop_da_camera, daemon=True)
    t.start()

    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    # use_reloader=False é vital quando se usa câmera com Flask, senão ele tenta ligar a câmera duas vezes e trava.
