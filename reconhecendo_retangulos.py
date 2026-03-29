import cv2
import numpy as np
import math

def pipeline_blackhat(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Erro: Imagem não encontrada.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_bh)

    _, mask_bh = cv2.threshold(blackhat, 53, 255, cv2.THRESH_BINARY)
    kernel_close = np.ones((2,2), np.uint8)
    mask_soldada = cv2.morphologyEx(mask_bh, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(mask_soldada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidatos = []
    out = img.copy()

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
        LIMITE_MIN_TRACO = 10 # Ignora cisco que o ratio achou que era linha

        # Adicionamos os limites de comprimento no IF principal
        if ratio > 2.0 and (LIMITE_MIN_TRACO <= linha_comprimento <= LIMITE_MAX_TRACO):

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

    pedras_aprovadas = []
    DISTANCIA_MINIMA = 34

    for cand in candidatos:
        cx1, cy1 = cand['centro']
        duplicata = False

        for aprovada in pedras_aprovadas:
            cx2, cy2 = aprovada['centro']
            dist = math.hypot(cx2 - cx1, cy2 - cy1)

            if dist < DISTANCIA_MINIMA:
                duplicata = True
                break

        if not duplicata:
            pedras_aprovadas.append(cand)

    print(f"Pedras únicas encontradas: {len(pedras_aprovadas)}")

    for d in pedras_aprovadas:
        box_traco = np.int32(cv2.boxPoints(d['rect_traco']))
        cv2.drawContours(out, [box_traco], 0, (255, 0, 0), 2)

        box_pedra = np.int32(cv2.boxPoints(d['rect_pedra']))
        cv2.drawContours(out, [box_pedra], 0, (0, 255, 0), 2)

    cv2.imshow("Mascara Limpa (Blackhat)", mask_soldada)
    cv2.imshow("Resultado Final Limpo", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# TESTE
# pipeline_blackhat("imagem_recortada.jpeg")
pipeline_blackhat("imagem.jpeg")
