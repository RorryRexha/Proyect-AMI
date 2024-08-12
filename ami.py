import tkinter as tk
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Cargar y Entrenar el Modelo (esto solo se ejecuta una vez)
def entrenar_modelo():
    # Crear el dataset de ejemplo (Pregunta y Respuesta)
    data = pd.DataFrame({
        'Pregunta': [
            '¿Qué hago si tengo fiebre alta?',
            '¿Cómo saber si tengo COVID-19?',
            '¿Es normal sentir dolor en el pecho?',
            '¿Qué puedo tomar para el dolor de cabeza?',
            '¿Qué hago si tengo diarrea?',
            '¿Cuánto tiempo dura la gripe?',
            '¿Qué debo hacer si me duele el estómago después de comer?',
            '¿Cómo puedo prevenir la hipertensión?',
            '¿Qué significa tener colesterol alto?',
            '¿Qué hago si tengo una reacción alérgica?',
            '¿Cómo puedo tratar una herida pequeña en casa?',
            '¿Qué debo hacer si tengo un dolor de garganta?',
            '¿Es peligroso tener presión arterial alta?',
            '¿Qué síntomas indican una posible fractura ósea?',
            '¿Cómo puedo prevenir el resfriado común?',
            '¿Qué hacer si tengo un dolor abdominal intenso?',
            '¿Cómo tratar una picadura de insecto en casa?',
            '¿Cuándo debería preocuparme por una tos persistente?',
            '¿Qué alimentos son buenos para mejorar la digestión?',
            '¿Cómo identificar si tengo una infección urinaria?'
        ],
        'Respuesta': [
            'Mantente hidratado y consulta a un médico si la fiebre persiste más de 3 días.',
            'Los síntomas comunes son fiebre, tos seca y fatiga. Realiza una prueba PCR para confirmarlo.',
            'El dolor en el pecho puede ser grave. Consulta a un médico inmediatamente si es severo.',
            'El paracetamol o ibuprofeno suelen ser efectivos. Consulta a un médico si persiste.',
            'Mantén una dieta blanda y bebe muchos líquidos. Si dura más de 2 días, consulta a un médico.',
            'La gripe suele durar entre 5 y 7 días, aunque la fatiga puede persistir durante más tiempo. Si los síntomas persisten o empeoran, consulta a un médico.',
            'Podría ser indigestión o una alergia alimentaria. Intenta identificar el alimento que causa el malestar y evita consumirlo. Si el dolor persiste, consulta a un médico.',
            'Mantén una dieta baja en sal, haz ejercicio regularmente, evita el tabaco y controla el estrés. Consultar a un médico para un plan específico es recomendable.',
            'El colesterol alto puede aumentar el riesgo de enfermedades cardíacas. Se recomienda una dieta baja en grasas saturadas, ejercicio regular, y en algunos casos, medicación.',
            'Si la reacción es leve, los antihistamínicos pueden ayudar. Si tienes dificultad para respirar o hinchazón grave, busca atención médica de inmediato.',
            'Lava la herida con agua y jabón, aplica un antiséptico y cúbrela con una venda limpia.',
            'Gargarear con agua salada y tomar líquidos calientes puede ayudar. Consulta a un médico si los síntomas persisten más de 3 días.',
            'La presión arterial alta puede aumentar el riesgo de enfermedades cardíacas y accidentes cerebrovasculares. Es importante controlarla y tratarla adecuadamente.',
            'Síntomas de una posible fractura incluyen dolor intenso, hinchazón, y dificultad para mover la parte afectada. Consulta a un médico para una evaluación.',
            'Mantén una buena higiene, evita el contacto cercano con personas enfermas, y considera vacunarte contra la gripe.',
            'Podría ser una indigestión o una condición más seria. Consulta a un médico si el dolor es severo, persistente o está acompañado de otros síntomas graves.',
            'Aplica una compresa fría para reducir la hinchazón y toma antihistamínicos si es necesario. Consulta a un médico si hay signos de una reacción alérgica severa.',
            'Una tos persistente puede ser un signo de una infección respiratoria o alergias. Consulta a un médico si la tos dura más de 2 semanas.',
            'Los alimentos ricos en fibra, como frutas, verduras y granos enteros, pueden ayudar a mejorar la digestión.',
            'Síntomas comunes incluyen dolor al orinar, necesidad frecuente de orinar, y orina turbia o con mal olor. Consulta a un médico para un diagnóstico adecuado.'
        ]
    })
    
    # Características y etiquetas
    X = data['Pregunta']
    y = data['Respuesta']

    # Vectorización TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)

    # División en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenamiento del modelo RandomForest
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Guardar el modelo entrenado
    joblib.dump(model, 'modelo_entrenado.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    # Evaluación del modelo
    evaluar_modelo(model, X_test, y_test, X, y)

# 2. Cargar el modelo para usarlo en predicciones
def cargar_modelo():
    model = joblib.load('modelo_entrenado.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

# 3. Función para responder preguntas usando el modelo cargado
def responder_pregunta(model, vectorizer, pregunta):
    pregunta_vectorizada = vectorizer.transform([pregunta])
    prediccion = model.predict(pregunta_vectorizada)
    return prediccion[0]

# 4. Evaluar el modelo
def evaluar_modelo(model, X_test, y_test, X, y):
    y_pred = model.predict(X_test)

    # Precisión del modelo
    precision = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {precision * 100:.2f}%")

    # Matriz de confusión
    matriz_confusion = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusión:\n", matriz_confusion)

    # Reporte de clasificación
    reporte = classification_report(y_test, y_pred)
    print("Reporte de Clasificación:\n", reporte)

    # Validación cruzada
    scores = cross_val_score(model, X, y, cv=5)
    print(f"Precisión media en validación cruzada: {np.mean(scores) * 100:.2f}%")

# 5. Crear la Interfaz Gráfica
def crear_interfaz_grafica():
    # Cargar el modelo entrenado y el vectorizador
    model, vectorizer = cargar_modelo()

    # Función para el botón "Enviar"
    def enviar_comando():
        comando = entrada.get()
        respuesta = responder_pregunta(model, vectorizer, comando)
        mostrar_respuesta(comando, respuesta)
        entrada.delete(0, tk.END)

    def mostrar_respuesta(comando, respuesta):
        historial.config(state=tk.NORMAL)
        historial.insert(tk.END, f"Tú: {comando}\n")
        historial.insert(tk.END, f"AMI: {respuesta}\n\n")
        historial.config(state=tk.DISABLED)
        historial.see(tk.END)  # Desplazarse al final del texto

    # Crear la ventana principal
    root = tk.Tk()
    root.title("Asistente Médico Inteligente (AMI)")
    root.geometry("800x450")

    # Configuración de colores
    bg_color = "#0057D8"
    panel_color = "#4C89FF"
    font_color = "white"

    # Estilo de la ventana
    root.configure(bg=bg_color)

    # Marco superior para el título y logo
    top_frame = tk.Frame(root, bg=bg_color, height=60)
    top_frame.pack(fill="x")

    title_label = tk.Label(top_frame, text="AMI", font=("Arial", 20), fg=font_color, bg=bg_color)
    title_label.pack(side="left", padx=20, pady=10)

    # Marco izquierdo para el historial y perfil
    left_frame = tk.Frame(root, bg=panel_color, width=200)
    left_frame.pack(side="left", fill="y")

    history_button = tk.Button(left_frame, text="History", font=("Arial", 16), fg=font_color, bg=panel_color, bd=0)
    history_button.pack(pady=20, padx=10, anchor="w")

    profile_button = tk.Button(left_frame, text="Profile", font=("Arial", 16), fg=font_color, bg=panel_color, bd=0)
    profile_button.pack(pady=20, padx=10, anchor="w")

    # Marco derecho para el chat
    chat_frame = tk.Frame(root, bg=panel_color)
    chat_frame.pack(expand=True, fill="both")

    chat_label = tk.Label(chat_frame, text="Chat", font=("Arial", 16), fg=font_color, bg=panel_color)
    chat_label.pack(anchor="nw", padx=20, pady=10)

    ami_logo = tk.Label(chat_frame, text="🤖\n❤️", font=("Arial", 60), fg=font_color, bg=panel_color)
    ami_logo.pack(expand=True)

    # Mostrar mensaje de bienvenida al iniciar
    welcome_message = "¡Bienvenido a AMI! Estoy aquí para ayudarte como te sientes el dia de Hoy."
    output_label = tk.Label(chat_frame, text=welcome_message, wraplength=400, bg=panel_color, fg=font_color, font=("Arial", 12))
    output_label.pack(pady=10)

    # Marco para el historial de mensajes
    historial = tk.Text(chat_frame, bg=panel_color, fg=font_color, font=("Arial", 12), wrap=tk.WORD, state=tk.DISABLED)
    historial.pack(expand=True, fill="both", padx=20, pady=10)

    input_frame = tk.Frame(chat_frame, bg=panel_color)
    input_frame.pack(side="bottom", fill="x", padx=20, pady=20)

    entrada = tk.Entry(input_frame, font=("Arial", 14), bd=0)
    entrada.pack(fill="x", padx=10, pady=5)  # Ajustar el espacio

    send_button = tk.Button(input_frame, text="Enviar", command=enviar_comando, bg=panel_color, fg=font_color)
    send_button.pack(side="right", padx=10)

    # Iniciar el bucle principal de la interfaz gráfica
    root.mainloop()

# 6. Ejecutar la interfaz gráfica y entrenar el modelo
if __name__ == "__main__":
    # Si el modelo aún no está entrenado, descomenta la siguiente línea
    #entrenar_modelo()

    crear_interfaz_grafica()
