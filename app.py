
import streamlit as st
import base64
import os
from dotenv import load_dotenv
import google.generativeai as genai
import tempfile
import PIL.Image

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("🔑 API-ключ Gemini не найден. Проверьте файл .env с переменной GOOGLE_API_KEY")
    st.stop()

try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"Ошибка конфигурации API Gemini: {str(e)}")
    st.stop()

sample_prompt = """Ты — практикующий врач и эксперт по интерпретации медицинских изображений, работающий в ведущем клиническом центре. Твоя задача — на основе предоставленного изображения провести медицинский анализ, определить возможные патологии или физиологические отклонения, и выдать структурированный отчёт.

Если изображение не относится к человеческому телу или не содержит диагностически значимой информации — прямо укажи это.

Структура ответа:

1) Описание изображения — кратко опиши, что именно изображено (например, КТ грудной клетки, МРТ мозга и т. п.).
2) Основные наблюдения — перечисли выявленные аномалии, структурные изменения, отклонения от нормы.
3) Предварительные выводы — укажи вероятные диагнозы или состояния, с осторожной формулировкой (например, «возможные признаки», «вероятно соответствует»).
4) Рекомендации — опиши, какие дополнительные обследования требуются (например, биопсия, анализы, динамическое наблюдение), а также общие рекомендации пациенту.
5) Отказ от ответственности — в конце обязательно добавь: «Проконсультируйтесь с врачом, прежде чем принимать какие-либо решения».
6) Если невозможно интерпретировать — честно укажи: «Невозможно определить на основе предоставленного изображения»."""


if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'result' not in st.session_state:
    st.session_state.result = None

def call_gemini_for_analysis(image_path, prompt=sample_prompt):
    """Функция для анализа изображения с помощью Gemini"""
    try:

        image = PIL.Image.open(image_path)

        try:

            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content([prompt, image])
            return response.text
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():

                st.warning("Превышена квота для модели gemini-1.5-flash. Пробуем использовать альтернативную модель.")
                try:
                    model = genai.GenerativeModel('gemini-pro-vision')
                    response = model.generate_content([prompt, image])
                    return response.text
                except Exception as e2:
                    raise Exception(f"Ошибка с альтернативной моделью: {str(e2)}") 
            else:
                raise e
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            return """
            ### ⚠️ Превышен лимит запросов API Gemini
            
            Вы превысили бесплатную квоту запросов к API Gemini. Варианты решения:
            
            1. Подождите некоторое время (обычно квота сбрасывается через 60 секунд или 24 часа)
            2. Зарегистрируйте платный аккаунт в Google AI Studio
            3. Используйте другой API-ключ
            
            Подробнее: [Лимиты API Gemini](https://ai.google.dev/gemini-api/docs/rate-limits)
            """
        else:
            return f"Произошла ошибка при анализе изображения: {error_msg}"

def chat_eli5(query):
    """Функция для упрощенного объяснения результатов"""
    try:
        eli5_prompt = "Вам необходимо объяснить следующую информацию пятилетнему ребенку. \n" + query
        
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(eli5_prompt)
            return response.text
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                st.warning("Превышена квота для модели gemini-1.5-flash. Пробуем использовать альтернативную модель.")
                try:
                    model = genai.GenerativeModel('gemini-pro')
                    response = model.generate_content(eli5_prompt)
                    return response.text
                except Exception as e2:
                    raise Exception(f"Ошибка с альтернативной моделью: {str(e2)}")
            else:
                raise e
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            return """
            ### ⚠️ Превышен лимит запросов API Gemini
            
            Вы превысили бесплатную квоту запросов к API Gemini. Пожалуйста, попробуйте позже или обновите свой план.
            """
        else:
            return f"Произошла ошибка при упрощении объяснения: {error_msg}"


st.title("Медицинский анализ с использованием Gemini")

with st.expander("О приложении"):
    st.write("Загрузите медицинское изображение для анализа с помощью Google Gemini.")

uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        st.session_state['filename'] = tmp_file.name

    st.image(uploaded_file, caption='Загруженное изображение')

# Кнопка обработки
if st.button('Анализировать изображение'):
    if 'filename' in st.session_state and os.path.exists(st.session_state['filename']):
        with st.spinner('Анализируем изображение...'):
            result = call_gemini_for_analysis(st.session_state['filename'])
            st.session_state['result'] = result
        
        if "Превышен лимит запросов API Gemini" in result:
            st.error(result)
            st.info("""
            ### Решения проблемы с квотой:
            
            1. Подождите несколько минут и попробуйте снова
            2. Создайте новый API-ключ в [Google AI Studio](https://makersuite.google.com/app/apikey)
            3. Перейдите на платный тариф
            """)
        else:
            st.markdown(result, unsafe_allow_html=True)
        
        if os.path.exists(st.session_state['filename']):
            os.unlink(st.session_state['filename'])

# Объяснение ELI5
if 'result' in st.session_state and st.session_state['result']:
    if "Превышен лимит запросов API Gemini" not in st.session_state['result']:
        st.info("Ниже у вас есть возможность получить упрощенное объяснение.")
        if st.radio("ELI5 - Объяснить как пятилетнему", ('Нет', 'Да')) == 'Да':
            with st.spinner('Готовим упрощенное объяснение...'):
                simplified_explanation = chat_eli5(st.session_state['result'])
                
                if "Превышен лимит запросов API Gemini" in simplified_explanation:
                    st.error(simplified_explanation)
                else:
                    st.markdown(simplified_explanation, unsafe_allow_html=True)